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
import pandas as pd
import pickle


load_dotenv()

weather_api_key=os.getenv("OPEN_WEATHER_API_KEY")

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


# get weather
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

def kelvin_to_fahrenheit(kelvin):
    return (kelvin - 273.15) * 9/5 + 32
    
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


def run_get_weather(user_prompt):
    # Step 1: send the conversation and available functions to the model
    messages=[
        {
            "role": "system",
            "content":  "Today is 17 April 2024,You are a function calling LLM that get the weather info at the provided datetime in ISO format with the provided period in a given location. if the provided period is not intger set the default period to 1"
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
            
            llm_response = run_get_weather(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""
        '''

        print ("[Zahaby] : I want to book an event on 2024-04-18 on 2:15 PM for 1 hour")    
        user_prompt = "I want to get the weather on 2024-04-18 on 2:15 PM for 1 hour in London"

        #print ("[Zahaby] : I want to book an event today on 11:15 PM for 1 hour")    
        #user_prompt = "I want to get the weather today on 11:15 PM for 1 hour in NY"

        #print ("[Zahaby] : I want to book an event next Wednsday on 2:15 PM for 1 hour")    
        #user_prompt = "I want to get the weather next Wednsday on 2:15 PM for 1 hour in CA"
        print (run_get_weather(user_prompt))

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())


