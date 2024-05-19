import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain

from langchain.chains.qa_with_sources import load_qa_with_sources_chain


from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

class LanguageModelProcessor:
    def __init__(self):
        self.chat_history = []

        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        
        #################################################
        ##### Loading data-sources
        text_loader_kwargs={'autodetect_encoding': True}
        loader_txt = DirectoryLoader("txt/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        documents_txt = loader_txt.load()
        #################################################
        #################################################
        ##### Embeddings Model setup
        ##### Vectorization

        # splitting doc
        
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)
        all_splits = text_splitter.split_documents(documents_txt)

        # specify embedding model (using huggingface sentence transformer)
#        embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        embedding_model_name = "all-MiniLM-L6-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)


        #document chunks and embiddings
        vectordb = FAISS.from_documents(all_splits, embeddings)

        self.retriever = vectordb.as_retriever()

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
       
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

#        self.conversation = LLMChain(
#            llm=self.llm,
#            prompt=PROMPT,
#            memory=self.memory,
#        )
        
        template = """
        You help everyone by answering questions, and improve your answers from previous answers in History.
        Don't try to make up an answer, if you don't know, just say that you don't know.
        Answer in the same language the question was asked.
        Answer in a way that is easy to understand.
        Do not say "Based on the information you provided, ..." or "I think the answer is...". Just answer the question directly in detail.
        
        History: {chat_history}
        
        Context: {context}
        
        Question: {question}
        Answer: 
        """

        PROMPT = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context", "question"]
        )

        tokenizer = AutoTokenizer.from_pretrained(self.llm)
        tokenizer.bos_token_id = 1  # Set beginning of sentence token id

        pipeline = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=tokenizer,
                use_cache=True,
                device_map="auto",
                max_length=1024,
                top_k=5,
                num_return_sequences=1,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
        )

        # specify the llm
        llm = HuggingFacePipeline(pipeline=pipeline)


        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            #verbose=False,
            memory=self.memory,
            get_chat_history=lambda h : h,
        )
      

 
    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory
        start_time = time.time()
    
        # Go get the response from the LLM
#        response = self.conversation.invoke({"text": text})
        response = self.conversation.invoke({'question': text, 'chat_history': self.memory.chat_memory})
        end_time = time.time()

#        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response['answer'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
#        print(f"LLM ({elapsed_time}ms): {response['text']}")
        print(f"LLM ({elapsed_time}ms): {response['answer']}")
#        return response['text']
        return response['answer']

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

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            tts.speak(llm_response)

            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())