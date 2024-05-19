## Groq Tools

Groq API endpoints support tool use for programmatic execution of specified operations through requests with explicitly defined operations. With tool use, Groq API model endpoints deliver structured JSON output that can be used to directly invoke functions from desired codebases.

### [Models](https://console.groq.com/docs/tool-use#models)

These following models powered by Groq all support tool use:

-   **llama3-70b**
-   **llama3-8b**
-   **llama2-70b**
-   **mixtral-8x7b**
-   **gemma-7b-it**

Parallel tool calling is enabled for both Llama3 models.

### [Use Cases](https://console.groq.com/docs/tool-use#use-cases)

-   **Convert natural language into API calls:**  Interpreting user queries in natural language, such as “What’s the weather in Palo Alto today?”, and translating them into specific API requests to fetch the requested information.
-   **Call external API:**  Automating the process of periodically gathering stock prices by calling an API, comparing these prices with predefined thresholds and automatically sending alerts when these thresholds are met.
-   **Resume parsing for recruitment:**  Analyzing resumes in natural language to extract structured data such as candidate name, skillsets, work history, and education, that can be used to populate a database of candidates matching certain criteria.

### [Example](https://console.groq.com/docs/tool-use#example)

```

from groq import Groq
import os
import json

client = Groq(api_key = os.getenv('GROQ_API_KEY'))
MODEL = 'mixtral-8x7b-32768'


# Example dummy function hard coded to return the score of an NBA game
def get_game_score(team_name):
    """Get the current score for a given NBA game"""
    if "warriors" in team_name.lower():
        return json.dumps({"game_id": "401585601", "status": 'Final', "home_team": "Los Angeles Lakers", "home_team_score": 121, "away_team": "Golden State Warriors", "away_team_score": 128})
    elif "lakers" in team_name.lower():
        return json.dumps({"game_id": "401585601", "status": 'Final', "home_team": "Los Angeles Lakers", "home_team_score": 121, "away_team": "Golden State Warriors", "away_team_score": 128})
    elif "nuggets" in team_name.lower():
        return json.dumps({"game_id": "401585577", "status": 'Final', "home_team": "Miami Heat", "home_team_score": 88, "away_team": "Denver Nuggets", "away_team_score": 100})
    elif "heat" in team_name.lower():
        return json.dumps({"game_id": "401585577", "status": 'Final', "home_team": "Miami Heat", "home_team_score": 88, "away_team": "Denver Nuggets", "away_team_score": 100})
    else:
        return json.dumps({"team_name": team_name, "score": "unknown"})

def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "You are a function calling LLM that uses the data extracted from the get_game_score function to answer questions around NBA game scores. Include the team and their opponent in your response."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_game_score",
                "description": "Get the score for a given NBA game",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team_name": {
                            "type": "string",
                            "description": "The name of the NBA team (e.g. 'Golden State Warriors')",
                        }
                    },
                    "required": ["team_name"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",  
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_game_score": get_game_score,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                team_name=function_args.get("team_name")
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content
    
user_prompt = "What was the score of the Warriors game?"
print(run_conversation(user_prompt))

```

### [Sequence of Steps](https://console.groq.com/docs/tool-use#sequence-of-steps)

-   **Initialize the API client**: Set up the Groq Python client with your API key and specify the model to be used for generating  [conversational responses](https://console.groq.com/docs/text-chat#streaming-a-chat-completion).
-   **Define the function and conversation parameters**: Create a user query and define a function (`get_current_score`) that can be called by the model, detailing its purpose, input parameters, and expected output format.
-   **Process the model’s request**: Submit the initial conversation to the model, and if the model requests to call the defined function, extract the necessary parameters from the model’s request and execute the function to get the response.
-   **Incorporate function response into conversation**: Append the function’s output to the conversation and a structured message and resubmit to the model, allowing it to generate a response that includes or reacts to the information provided by the function call.

### [Tools Specifications](https://console.groq.com/docs/tool-use#tools-specifications)

-   `tools`: an array with each element representing a tool
    -   `type`: a string indicating the category of the tool
    -   `function`: an object that includes:
        -   `description`  - a string that describes the function’s purpose, guiding the model on when and how to use it
        -   `name`: a string serving as the function’s identifier
        -   `parameters`: an object that defines the parameters the function accepts

### [Tool Choice](https://console.groq.com/docs/tool-use#tool-choice)

-   `tool_choice`: A parameter that dictates if the model can invoke functions.
    -   `auto`: The default setting where the model decides between sending a text response or calling a function
    -   `none`: Equivalent to not providing any tool specification; the model won't call any functions
-   Specifying a Function:
    -   To mandate a specific function call, use  `{"type": "function", "function": {"name":"get_financial_data"}}`
    -   The model is constrained to utilize the function named

### [Known limitations](https://console.groq.com/docs/tool-use#known-limitations)

-   Parallel tool use is disabled because of limitations of the Mixtral model. The endpoint will always return at most a single  `tool_call`  at a time.

ref: https://console.groq.com/docs/tool-use

