- Here we are utilizing same stack (groq+deepgram) to build a calendar scheduler POC. 
- We are integrate Gmail API using Gmail API-Key to access google calendar.
- We are using groq tool to perform actions based on the conversation.

# Code Implementation
```

def run_conversation_book(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "Today is 17 April 2024,You are a function calling LLM that Book an event on the calendar at the provided datetime in ISO format with the provided period. if the provided period is not intger set the default period to 1"
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
                "name": "book_event",
                "description": "Set calendar events",
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
            "book_event": book_event,
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
"content": "Today is 17 April 2024,You are a function calling LLM that Book an event on the calendar at the provided datetime in ISO format with the provided period. if the provided period is not intger set the default period to 1"

- The model is not aware of the current date and time, so I am informing the model about today's date.

- I am also telling the model via the prompt about the format of the datetime and time, so I am limiting any date-time conversion to a specific format. 

- and finally, I am telling the prompt to set a default value of 1 hour for the meeting time in case of not provided.

- Then I am defining groq tool template
```
    tools = [
        {
            "type": "function",
            "function": {
                "name": "book_event",
                "description": "Set calendar events",
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
- Telling the model that the action function is : book_event function.
- Finally pass the extracted params to the function.

## book_event function and Gmail APIs

```
# Create Calendar event
def book_event(str_datetime , period):
    #if period == "Custom":
    #    hours = 0.5
    #else:
    #    hours = int(re.search(r'\d+', period).group())
    #hours = int(re.search(r'\d+', period).group())
    hours = period

    # Authenticate Google Calendar API
    oauth2_client_secret_file = './cred.json'
    scopes = ['https://www.googleapis.com/auth/calendar']
    service = authenticate_google(scopes=scopes, oauth2_client_secret_file=oauth2_client_secret_file)

    # Get email-ids of all subscribed calendars
    calendars_result = service.calendarList().list().execute()

    calendars = calendars_result.get('items', [])
    
    print(str_datetime)
    str_datetime = str_datetime.replace ('AM','').replace('PM','').replace('T',' ')
    if (len(str_datetime.split(':'))) > 2:
        str_datetime = str_datetime[:-3]

    print(str_datetime)
    
    # Feature 3: Insert an event
    event = {
        'summary': 'AI-Reserved Meeting',
        'location': 'Zoom meeting',
        'description': 'A meeting scheduled by AI.',
        'start': {
            'dateTime': (datetime.strptime(str_datetime, '%Y-%m-%d %H:%M')).isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': (datetime.strptime(str_datetime, '%Y-%m-%d %H:%M') + timedelta(hours=hours)).isoformat(),
            'timeZone': 'America/Los_Angeles',
        },
    }
    created_event = service.events().insert(calendarId="zahaby@gmail.com", body=event).execute()
    print(f"Created event: {created_event['id']}")
    return json.dumps({"Created event": created_event['description']})
```

## TODO:
- Check the reservation date-time first. (by calling calendar-list API). if reserved suggest other free slots. 
- This is a stateless calendar reservation POC, doesn't handle any type of conversation or stateful flow. Need to put this POC in a full context. 
- Handle exceptions, as there is a big window for different error scenarios. 

### useful links:

https://developers.google.com/gmail/api/quickstart/python#authorize_credentials_for_a_desktop_application

https://python.langchain.com/docs/integrations/toolkits/gmail/

https://developers.google.com/calendar/api/quickstart/python#authorize_credentials_for_a_desktop_application
