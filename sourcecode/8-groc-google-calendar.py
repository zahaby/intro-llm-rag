import shutil
import asyncio
import subprocess
import requests
import time
import os
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from dotenv import load_dotenv
import re
#import datetime
from datetime import timedelta,datetime

from groq import Groq
import os
import json
from google_utils import authenticate_google, query_google_calendar_api
import pandas as pd
import pickle


load_dotenv()

client = Groq(api_key = "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")
#MODEL = 'mixtral-8x7b-32768'
MODEL = 'llama3-70b-8192'


class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()


def convert24(str1): 
    in_time = datetime.strptime(str1, "%I:%M %p")
    return datetime.strftime(in_time, "%H:%M")


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


# Get Calendar events
def list_calendar(day):
    # Authenticate Google Calendar API
    oauth2_client_secret_file = './cred.json'
    scopes = ['https://www.googleapis.com/auth/calendar']
    service = authenticate_google(scopes=scopes, oauth2_client_secret_file=oauth2_client_secret_file)

    # Get email-ids of all subscribed calendars
    calendars_result = service.calendarList().list().execute()

    calendars = calendars_result.get('items', [])
    emails = [c['id'] for c in calendars]
    #print(emails)
    calendar_results = service.events().list(
        calendarId="zahaby@gmail.com", 
        timeMin=day+'T00:00:00Z',
        timeMax=day+'T23:59:00Z',
        maxResults=10, 
        singleEvents=True,
        orderBy='startTime').execute()

    #print (len(calendar_results.get('items', [])))
    if len(calendar_results.get('items', [])) > 0:
        df = pd.DataFrame(calendar_results.get('items', []))
        
        #columns_keep = ["summary", "creator", "start", "end", "attendees", "location", "id"]
        columns_keep = ["summary", "creator", "start", "end", "attendees", "id"]
        df = df[columns_keep]
        df = df.rename(columns={"summary": "name"})

        df["timeZone"] = df["start"].apply(lambda x : x.get("timeZone", "Europe/Berlin"))
        # API delivers entries with dateTime or date, we want a single type in the column
        df["start"] = df["start"].apply(lambda x : pd.to_datetime(x.get("dateTime", x.get("date")), utc=True))
        df["end"] = df["end"].apply(lambda x : pd.to_datetime(x.get("dateTime", x.get("date")), utc=True))
        df["duration"] = df.end - df.start

        #print(df)
        #df.head(1)

        #return df_test.shape
        
        #return json.dumps({"Day": day, "event1": "rooma birthday", "event2": "zahaby birthday"})
        return df.to_json()
    else: 
        return json.dumps({"events": "no events that day"})

def run_conversation(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content": "You are a function calling LLM that uses the data extracted from the list_calendar function to answer questions around calendar events. Include the calendar's day in your response."
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
                "name": "list_calendar",
                "description": "Get calendar events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "day": {
                            "type": "string",
                            "description": "The calendar day",
                        }
                    },
                    "required": ["day"],
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
            "list_calendar": list_calendar,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print (function_args)
            
            function_response = function_to_call(
                day=function_args.get("day")
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

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)
    
transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return


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

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
        '''
        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = run_conversation_book(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""
        '''

        #user_prompt = "Is there any event on 2023-10-15?"
        #print(run_conversation(user_prompt))
        #user_prompt = input()

        print ("[Zahaby] : I want to book an event on 2024-04-18 on 2:15 PM for 1 hour")    
        user_prompt = "I want to book an event on 2024-04-18 on 2:15 PM for 1 hour"

        #print ("[Zahaby] : I want to book an event today on 11:15 PM for 1 hour")    
        #user_prompt = "I want to book an event today on 11:15 PM for 1 hour"

        #print ("[Zahaby] : I want to book an event next Wednsday on 2:15 PM for 1 hour")    
        #user_prompt = "I want to book an event next Wednsday on 2:15 PM for 1 hour"
        print (run_conversation_book(user_prompt))

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())


