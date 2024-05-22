- Here we are utilizing same stack (groq+deepgram) to get the weather details in a specific location on a specific date for a period. 
- We are integrate openweathermap using openweathermap to access the weather service.
- We are using groq tool to perform actions based on the conversation.

# Code Implementation
```

def run_conversation_book(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "Today is 17 April 2024,You are a function calling LLM that get the weather info at the provided datetime in ISO format with the provided period in a given location. if the provided period is not intger set the default period to 1"
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
                "name": "get_weather",
                "description": "Get Weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "str_datetime": {
                            "type": "string",
                            "description": "The event date and time",
                        },
                        "period": {
                            "type": "integer",
                            "description": "The event period",
                        },
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["str_datetime", "period","location"],
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
            "get_weather": get_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print (function_args)
            
            function_response = function_to_call(
                str_datetime=function_args.get("str_datetime"),
                period=function_args.get("period") 
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


print ("[AI] : hi, please pickup a date, time for the meeting")    

```
- defining the prompt here is very important:
"content": "Today is 17 April 2024,You are a function calling LLM that get the weather info at the provided datetime in ISO format with the provided period in a given location. if the provided period is not intger set the default period to 1"

- The model is not aware of the current date and time, so I am informing the model about today's date.

- I am also telling the model via the prompt about the format of the datetime and time, so I am limiting any date-time conversion to a specific format. 

- and finally, I am telling the prompt to set a default value of 1 hour for the meeting time in case of not provided.

- Then I am defining groq tool template
```
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get Weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "str_datetime": {
                            "type": "string",
                            "description": "The event date and time",
                        },
                        "period": {
                            "type": "integer",
                            "description": "The event period",
                        }
                    },
                    "required": ["str_datetime", "period"],
                },
            },
        }
    ]
```
- make sure to define the correct data types.
- Here, I am informing the model that I am expecting two mandatory params (ste_datetime, period) with types (string, integer), respectively. 
- Telling the model that the action function is : get_weather function.
- Finally pass the extracted params to the function.

## get_weather function and Gmail APIs

```
# Get Weather
def get_weather(str_datetime , period, location):
    #if period == "Custom":
    #    hours = 0.5
    #else:
    #    hours = int(re.search(r'\d+', period).group())
    #hours = int(re.search(r'\d+', period).group())
    hours = period

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    location = location.lower()

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            weather_info = {
                'temperature': kelvin_to_fahrenheit(data['main']['temp']),
            }
            return json.dumps(f"temperature_in_{location}: {weather_info['temperature']}")
        else:
            return {'error': f"Error {response.status_code}: {data['message']}"}

    except Exception as e:
        return {'error': f"An error occurred: {str(e)}"}
```

## TODO:
- This is a stateless POC, doesn't handle any type of conversation or stateful flow. Need to put this POC in a full context. 
- Handle exceptions, as there is a big window for different error scenarios. 

### useful links:

https://medium.com/@stevenoluwaniyi/how-to-call-external-functions-using-openai-ais-7650589a127f

https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application

https://python.langchain.com/docs/integrations/toolkits/gmail/

https://developers.google.com/calendar/api/quickstart/python#authorize_credentials_for_a_desktop_application
