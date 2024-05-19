- In this implementation, we are using remote calls for (ASR, TTS, and LLM).
- We are utilizing Deepgram for ASR, TTS and qroq for Inference.
- We need to create an API-key for qroq here: https://console.groq.com/keys
- We need to create an API-key for deepgram as per doc: https://developers.deepgram.com/docs/create-additional-api-keys

# Code Implementation

**Install required dependencies**
```
import  asyncio
from  dotenv  import  load_dotenv
import  shutil
import  subprocess
import  requests
import  time
import  os
from  langchain_community.vectorstores  import FAISS
from  langchain_community.document_loaders  import  DirectoryLoader
from  langchain.document_loaders  import TextLoader
from  langchain.text_splitter  import  RecursiveCharacterTextSplitter, haracterTextSplitter
from  langchain.embeddings  import HuggingFaceEmbeddings
from  langchain.prompts  import  PromptTemplate
from  langchain.chains  import  ConversationalRetrievalChain, ConversationChain
from  langchain.chains.qa_with_sources  import  load_qa_with_sources_chain
from  langchain.chains  import  create_history_aware_retriever
from  langchain.chains  import  create_retrieval_chain
from  langchain.chains.combine_documents  import  create_stuff_documents_chain
from  langchain_core.messages  import  HumanMessage
from  langchain_core.prompts  import  ChatPromptTemplate
from  langchain_groq  import  ChatGroq
from  langchain_openai  import  ChatOpenAI
from  langchain.memory  import  ConversationBufferMemory, VectorStoreRetrieverMemory
from  langchain.prompts  import (
ChatPromptTemplate,
MessagesPlaceholder,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate,
)
from  langchain.chains  import  LLMChain
from  langchain_core.runnables  import  RunnablePassthrough
from  langchain_core.output_parsers  import  StrOutputParser
from  transformers  import  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from  langchain.llms  import HuggingFacePipeline
from  langchain_core.chat_history  import  BaseChatMessageHistory
from  langchain_community.chat_message_histories  import  ChatMessageHistory
from  langchain_core.runnables.history  import  RunnableWithMessageHistory
import  sys
from  deepgram  import (
DeepgramClient,
DeepgramClientOptions,
LiveTranscriptionEvents,
LiveOptions,
Microphone,
)
```
**Import the ChatGroq class and initialize it with a model:**

```
self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
```
This code snippet establishes a Groq client object to interact with the Groq API. It begins by retrieving the API key from an environment variable named GROQ_API_KEY and passes it to the argument api_key. Subsequently, the API key initializes the Groq client object, enabling API calls to the Large Language Models within Groq Servers.

**load docs for rag same way as listed in the previous implementations**

**use same embedding model sentence-transformers/all-mpnet-base-v2**

**user same vector db FAISS**

# Add chat history

In many Q&A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of “memory” of past questions and answers, and some logic for incorporating those into its current thinking.

In this guide we focus on  **adding logic for incorporating historical messages.**  Further details on chat history management is  [covered here](https://python.langchain.com/docs/expression_language/how_to/message_history/).

We’ll work off of the Q&A app we built over the  [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)  blog post by Lilian Weng in the  [Quickstart](https://python.langchain.com/docs/use_cases/question_answering/quickstart/). We’ll need to update two things about our existing app:

1.  **Prompt**: Update our prompt to support historical messages as an input.
2.  **Contextualizing questions**: Add a sub-chain that takes the latest user question and reformulates it in the context of the chat history. This is needed in case the latest question references some context from past messages. For example, if a user asks a follow-up question like “Can you elaborate on the second point?”, this cannot be understood without the context of the previous message. Therefore we can’t effectively perform retrieval with a question like this.

## Contextualizing the question[​](https://python.langchain.com/docs/use_cases/question_answering/chat_history/#contextualizing-the-question "Direct link to Contextualizing the question")

First we’ll need to define a sub-chain that takes historical messages and the latest user question, and reformulates the question if it makes reference to any information in the historical information.

We’ll use a prompt that includes a  `MessagesPlaceholder`  variable under the name “chat_history”. This allows us to pass in a list of Messages to the prompt using the “chat_history” input key, and these messages will be inserted after the system message and before the human message containing the latest question.

Note that we leverage a helper function  [create_history_aware_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html)  for this step, which manages the case where  `chat_history`  is empty, and otherwise applies  `prompt | llm | StrOutputParser() | retriever`  in sequence.

`create_history_aware_retriever`  constructs a chain that accepts keys  `input`  and  `chat_history`  as input, and has the same output schema as a retriever.
```
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```
This chain prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of the conversation.

## Chain with chat history[​](https://python.langchain.com/docs/use_cases/question_answering/chat_history/#chain-with-chat-history "Direct link to Chain with chat history")

And now we can build our full QA chain.

Here we use  [create_stuff_documents_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html)  to generate a  `question_answer_chain`, with input keys  `context`,  `chat_history`, and  `input`– it accepts the retrieved context alongside the conversation history and query to generate an answer.

We build our final  `rag_chain`  with  [create_retrieval_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html). This chain applies the  `history_aware_retriever`  and  `question_answer_chain`  in sequence, retaining intermediate outputs such as the retrieved context for convenience. It has input keys  `input`  and  `chat_history`, and includes  `input`,  `chat_history`,  `context`, and  `answer`  in its output.

```
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

## Code Flow:

![](https://python.langchain.com/assets/images/conversational_retrieval_chain-5c7a96abe29e582bc575a0a0d63f86b0.png)

Here we’ve gone over how to add application logic for incorporating historical outputs, but we’re still manually updating the chat history and inserting it into each input. In a real Q&A application we’ll want some way of persisting chat history and some way of automatically inserting and updating it.

For this we can use:

-   [BaseChatMessageHistory](https://python.langchain.com/docs/modules/memory/chat_messages/): Store chat history.
-   [RunnableWithMessageHistory](https://python.langchain.com/docs/expression_language/how_to/message_history/): Wrapper for an LCEL chain and a  `BaseChatMessageHistory`  that handles injecting chat history into inputs and updating it after each invocation.

For a detailed walkthrough of how to use these classes together to create a stateful conversational chain, head to the  [How to add message history (memory)](https://python.langchain.com/docs/expression_language/how_to/message_history/)  LCEL page.

Below, we implement a simple example of the second option, in which chat histories are stored in a simple dict.

Full Code:

```
 ### Contextualize question ###
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )


        ### Answer question ###
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]


        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        start_time = time.time()
    
        # Go get the response from the LLM
        response = self.conversational_rag_chain.invoke({"input": text},config={"configurable": {"session_id": "abc123"}},)
        end_time = time.time()
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response["answer"]}")
        self.chat_history.extend([HumanMessage(content=text), response["answer"]])
        return response["answer"]
```
## ASR with Deepgram:

```
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
```

`TranscriptCollector`  is a calls that is used to collect and manage parts of a transcript. A transcript is typically a written or printed record of what was said during a conversation, meeting, or interview.

Here's a breakdown of the class:

-   `__init__`: This is a special method that is automatically called when an object of the class is created. In this case, it calls the  `reset`  method, which initializes an empty list  `transcript_parts`.
    
-   `reset`: This method resets the  `transcript_parts`  list to an empty list.
    
-   `add_part`: This method adds a new part to the  `transcript_parts`  list.
    
-   `get_full_transcript`: This method returns the full transcript by joining all the parts in the  `transcript_parts`  list with a space.
    

The last line creates an instance of the  `TranscriptCollector`  class and assigns it to the variable  `transcript_collector`.

```
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
```

This code is an implementation of a speech-to-text system using the Deepgram API. Here's a breakdown of what the code does:

1.  Creates an instance of  `TranscriptCollector`.
    
2.  The  `get_transcript`  function is an asynchronous function that starts a transcription process. It uses the Deepgram API to listen to the user's microphone and transcribe the audio in real-time.
    
3.  The function sets up a Deepgram client with a configuration that keeps the connection alive.
    
4.  It then starts a live transcription session with the  `listen.asynclive.v("1")`  method. This method returns a  `dg_connection`  object that is used to send and receive data.
    
5.  The function defines an  `on_message`  function that is called whenever a new message is received from the Deepgram API. This function is responsible for processing the transcription data.
    
6.  The  `on_message`  function checks if the received message is the final part of a sentence. If it is, it adds the sentence to the  `TranscriptCollector`  and resets it. It then prints the full transcript and calls the  `callback`  function with the full transcript.
    
7.  The function then starts the transcription process by calling  `dg_connection.start(options)`. This method starts the transcription process with the specified options.
    
8.  It then opens a microphone stream and starts it.
    
9.  The function waits for the transcription to complete by calling  `transcription_complete.wait()`. This method blocks until the transcription is complete.
    
10.  After the transcription is complete, the function waits for the microphone to close and then finishes the transcription process by calling  `dg_connection.finish()`.
    
11.  If any exceptions occur during the transcription process, the function catches them and prints an error message.
    

The  `callback`  function is not defined in this code snippet, but it is likely a function that is called when the transcription is complete. It is passed the full transcript as an argument.


## TTS with Deepgram:

```
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
```

`TextToSpeech`  is a class that uses the Deepgram API to convert text to speech. Here's a breakdown of the code:

**Class variables**

-   `DG_API_KEY`: The Deepgram API key, set using the  `os.getenv`  function to retrieve an environment variable named  `DEEPGRAM_API_KEY`.
-   `MODEL_NAME`: The name of the Deepgram model to use for text-to-speech conversion, set to  `"aura-helios-en"`  (English).

**`is_installed`  method**

-   This method checks if a given library (e.g.,  `ffplay`) is installed on the system.
-   It uses the  `shutil.which`  function to search for the executable in the system's PATH.
-   If the executable is found, the method returns  `True`, otherwise it returns  `False`.

**`speak`  method**

-   This method takes a  `text`  parameter and converts it to speech using the Deepgram API.
-   It checks if  `ffplay`  is installed using the  `is_installed`  method. If not, it raises a  `ValueError`.
-   It sets the Deepgram API URL, headers, and payload for the request.
-   It uses the  `requests`  library to send a POST request to the Deepgram API with the text to be converted to speech.
-   It uses the  `ffplay`  command-line tool to play the audio stream.
-   It records the time before sending the request and calculates the time to first byte (TTFB) by measuring the time between sending the request and receiving the first byte of the audio stream.
-   It writes the audio stream to the  `ffplay`  process's stdin and flushes the buffer.
-   Finally, it closes the stdin stream and waits for the  `ffplay`  process to finish.

**Notes**

-   The  `speak`  method assumes that  `ffplay`  is installed and available on the system.
-   The  `MODEL_NAME`  variable can be changed to use a different Deepgram model.
-   The  `DG_API_KEY`  variable should be set to a valid Deepgram API key.
-   The  `speak`  method returns no value, but it prints the TTFB time to the console.

Overall, this code provides a simple way to convert text to speech using the Deepgram API and play the audio stream using  `ffplay`.

## Manage the Conversation:

```

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
```

1.  The  `ConversationManager`  class is initialized with an empty string  `transcription_response`  and an instance of  `LanguageModelProcessor`  (LLM) for language processing.
    
2.  The  `main`  method is an asynchronous function that runs indefinitely until the user says "goodbye".
    
3.  Inside the  `main`  method, it calls the  `get_transcript`  function with a callback function  `handle_full_sentence`. This function is called whenever a full sentence is transcribed.
    
4.  The  `handle_full_sentence`  function updates the  `transcription_response`  with the full sentence.
    
5.  The code then checks if the  `transcription_response`  contains the word "goodbye" (case-insensitive). If it does, the loop breaks and the program exits.
    
6.  If the  `transcription_response`  does not contain "goodbye", the code processes the  `transcription_response`  using the  `LanguageModelProcessor`  (LLM) to generate a response.
    
7.  The LLM response is then converted to speech using a  `TextToSpeech`  object and spoken to the user.
    
8.  Finally, the  `transcription_response`  is reset to an empty string for the next iteration of the loop.
    
