- We use 2XA4000 GPUs with low memory and the Mistral 7B model in this experiment.
- The first big challenge is to utilize the model to run with low memory. For this we had to use quantization.

- We use 2XA4000 GPUs with low memory and the Mistral 7B model in this experiment.
- The first big challenge is to utilize the model to run with low memory. For this we had to use quantization.

# Code Implementation

**Install required dependencies**
```
# import dependencies
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, AutoConfig, TextStreamer, TextIteratorStreamer

import os
import gradio as gr

from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader #for pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.document_loaders import UnstructuredURLLoader #for html
from langchain_community.vectorstores import FAISS
from IPython.display import Audio, display
from gtts import gTTS
from io import BytesIO
import base64
import time

from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
```

**define template:**
```
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise. 
Question: question 
Context: context 
Answer:""" 

prompt = PromptTemplate.from_template(template)
```
This code is defining a template for a question-answering system using the  `langchain/prompts`  library. The template is a string that contains placeholders for a question and context. The system will use this context to answer the given question.

Here's a breakdown of the template string:

1.  `You are an assistant for question-answering tasks.`  - This line introduces the purpose of the system.
2.  `Use the following pieces of retrieved context to answer the question.`  - This line instructs the system to use the provided context to generate an answer.
3.  `If you don't know the answer, just say that you don't know.`  - This line sets an expectation for the system to respond honestly when it can't answer a question.
4.  `Use two sentences maximum and keep the answer concise.`  - This line encourages the system to provide brief and to-the-point answers.
5.  `Question: question`  - This placeholder will be replaced with the actual question at runtime.
6.  `Context: context`  - This placeholder will be replaced with the actual context at runtime.
7.  `Answer:`  - This line separates the context from the system-generated answer.

The  `prompt`  variable is created using the  `PromptTemplate.from_template()`  function, which converts the template string into a  `PromptTemplate`  object. This object can then be used to generate prompts for the question-answering system.

**ASR:**
```
import whisper
model_whisper = whisper.load_model("base")

def transcribe(audio):

    start_time = time.time()

    language = 'en'

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)

    # detect the spoken language
    _, probs = model_whisper.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model_whisper, mel, options)

    print("---ASR: %s seconds ---" % (time.time() - start_time))

    return result.text
#################################################
```
This code is using the  `whisper`  library for automatic speech recognition (ASR) to transcribe audio files into text.

1.  `import whisper`  - Import the  `whisper`  library, which is a Python library for speech recognition.
2.  `model_whisper = whisper.load_model("base")`  - Load the pre-trained base model from the  `whisper`  library.
3.  `def transcribe(audio):`  - Define a function called  `transcribe`  that takes an audio file as input.
4.  `start_time = time.time()`  - Record the start time for measuring the transcription time.
5.  `language = 'en'`  - Set the language for the transcription to English.
6.  `audio = whisper.load_audio(audio)`  - Load the audio file using the  `whisper.load_audio()`  function.
7.  `audio = whisper.pad_or_trim(audio)`  - Pad or trim the audio to fit a length of 30 seconds.
8.  `mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)`  - Convert the audio into a log-Mel spectrogram and move it to the same device as the model.
9.  `_, probs = model_whisper.detect_language(mel)`  - Detect the language spoken in the audio using the  `detect_language()`  function.
10.  `options = whisper.DecodingOptions()`  - Create an instance of the  `DecodingOptions`  class for decoding the audio.
11.  `result = whisper.decode(model_whisper, mel, options)`  - Decode the audio using the  `decode()`  function, which returns a  `DecodingResult`  object.
12.  `print("---ASR: %s seconds ---" % (time.time() - start_time))`  - Calculate and print the time taken for the transcription.
13.  `return result.text`  - Return the transcribed text from the  `DecodingResult`  object.

The  `transcribe()`  function can be called by passing an audio file path as an argument to transcribe the audio into text.

**utilize the two GPUs**
```

device_ids = [0, 1]  # Modify this list according to your GPU configuration
primary_device = f'cuda:{device_ids[1]}'  # Primary device
torch.cuda.set_device(primary_device)

```
This code sets the primary GPU device for PyTorch to use for computations.

1.  `device_ids = [0, 1]`  - Define a list of GPU device IDs available for use. In this case, both devices with IDs 0 and 1 are included. Modify this list according to your GPU configuration.
2.  `primary_device = f'cuda:{device_ids[1]}'`  - Set the primary device to the second GPU in the list (index 1). In this case, it is set to 'cuda:1' assuming that the GPU at index 1 is available.
3.  `torch.cuda.set_device(primary_device)`  - Set the primary device for PyTorch to use for computations using the  `torch.cuda.set_device()`  function.

After running this code, PyTorch will use the specified GPU as the primary device for computations. If you have multiple GPUs and want to utilize them for parallel processing, you can modify the  `device_ids`  list and the  `primary_device`  assignment accordingly.

**initialize tokenizer:**

```
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```
This code initializes a tokenizer for a pre-trained model and sets padding configurations.

1.  `tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)`  - Initialize a tokenizer for a pre-trained model using the  `AutoTokenizer.from_pretrained()`  function. The  `trust_remote_code=True`  argument allows the function to download and execute pre-trained model scripts from a remote location if necessary.
2.  `tokenizer.pad_token = tokenizer.eos_token`  - Set the padding token for the tokenizer to be the end-of-sentence token. This ensures that sequences are padded with the appropriate token during tokenization.
3.  `tokenizer.padding_side = "right"`  - Set the padding side to the right. This means that when sequences are padded, the padding tokens will be added to the right side of the sequence.

After running this code, you will have a tokenizer object configured for a pre-trained model with padding settings applied. This tokenizer can be used for tokenizing input sequences and preparing them for input into a pre-trained model.

**quantization:**
```
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)
```
This code initializes a  `BitsAndBytesConfig`  object for configuring quantization settings for a model.

1.  `BitsAndBytesConfig(...)`  - Create a  `BitsAndBytesConfig`  object to configure quantization settings for a model.
2.  `load_in_4bit=True`  - Enable loading the model in 4-bit precision.
3.  `bnb_4bit_use_double_quant=True`  - Enable double quantization for 4-bit models.
4.  `bnb_4bit_quant_type="nf4"`  - Set the quantization type to "nf4", which stands for "neural-fused 4-bit".
5.  `bnb_4bit_compute_dtype=torch.bfloat16`  - Set the compute dtype to  `torch.bfloat16`  for 4-bit models.

After running this code, you will have a  `BitsAndBytesConfig`  object with the specified quantization settings. This object can be used for configuring a model to use 4-bit quantization during training or inference.

**initialize LLM:*

```
model_name='mistralai/Mistral-7B-Instruct-v0.2'
model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=config,config=model_config,device_map='auto')
```
This code initializes a pre-trained language model with quantization settings.

1.  `model_name='mistralai/Mistral-7B-Instruct-v0.2'`  - Specify the name of the pre-trained model.
2.  `model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)`  - Initialize a model configuration for the pre-trained model using the  `AutoConfig.from_pretrained()`  function.
3.  `model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=config,config=model_config,device_map='auto')`  - Initialize the pre-trained model using the  `AutoModelForCausalLM.from_pretrained()`  function. The  `quantization_config`  argument is set to the  `config`  object created earlier, which enables quantization for the model. The  `config`  argument is set to the  `model_config`  object, which specifies the model's configuration. The  `device_map`  argument is set to 'auto', which automatically maps the model's layers to the available GPUs.

After running this code, you will have a pre-trained language model initialized with the specified quantization settings. This model can be used for natural language processing tasks such as text generation, question answering, and more.

**Use both GPUs together:**
```
# Move model to GPUs
model = torch.nn.DataParallel(model, device_ids=device_ids)
```

This code moves the model to the specified GPUs using PyTorch's  `DataParallel`  module.

1.  `model = torch.nn.DataParallel(model, device_ids=device_ids)`  - Wrap the model with PyTorch's  `DataParallel`  module. This module replicates the model across the specified GPUs and handles data distribution and synchronization during training.
    -   `model`  - The model to be parallelized.
    -   `device_ids`  - A list of GPU IDs to use for parallel processing.

After running this code, the model will be parallelized across the specified GPUs, allowing for efficient data distribution and computation during training.

**initialize pipeline:**

```
#streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
streamer  = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

pipeline = pipeline(task='text-generation',
        model=model.module,
        tokenizer=tokenizer,
#        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1500,
        do_sample=False,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        streamer = streamer
)

llm = HuggingFacePipeline(pipeline=pipeline)
```

This code creates a Hugging Face pipeline for text generation using a pre-trained language model.

1.  `streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)`  - Initialize a  `TextIteratorStreamer`  object for streaming tokenized input and output.
    -   `tokenizer`  - The tokenizer associated with the pre-trained model.
    -   `skip_prompt=True`  - Skip the prompt token during streaming.
    -   `skip_special_tokens=True`  - Skip special tokens during streaming.
2.  `pipeline = pipeline(task='text-generation', ...)`  - Initialize a Hugging Face pipeline for text generation.
    -   `task='text-generation'`  - Specify the task for the pipeline.
    -   `model=model.module`  - The pre-trained model to use for text generation.
    -   `tokenizer=tokenizer`  - The tokenizer associated with the pre-trained model.
    -   `repetition_penalty=1.1`  - Apply a repetition penalty to discourage repeating the same phrases.
    -   `return_full_text=True`  - Return the full text instead of individual tokens.
    -   `max_new_tokens=1500`  - Set the maximum number of new tokens to generate.
    -   `do_sample=False`  - Disable sampling and use greedy decoding.
    -   `pad_token_id = tokenizer.eos_token_id`  - Set the padding token ID to the end-of-sentence token ID.
    -   `eos_token_id = tokenizer.eos_token_id`  - Set the end-of-sentence token ID.
    -   `streamer = streamer`  - Set the  `TextIteratorStreamer`  object for streaming tokenized input and output.
3.  `llm = HuggingFacePipeline(pipeline=pipeline)`  - Wrap the pipeline in a  `HuggingFacePipeline`  object for easier use.

After running this code, you will have a Hugging Face pipeline for text generation using a pre-trained language model. You can use the  `llm`  object to generate text based on input prompts.

**loading RAG data:**

```
text_loader_kwargs={'autodetect_encoding': True}
loader_txt = DirectoryLoader("txt/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
documents_txt = loader_txt.load()

if 1==0:
# load pdfs
    loader_pdfs = PyPDFDirectoryLoader('pdfs/')
    documents_pdfs = loader_pdfs.load()
#print(documents)

if 1==0:
    urls = [
        "https://www.someurl.com/some-html/",
        "https://www.someurl.com/some-content/",
        "https://www.someurl.com/file-and-docs/"
    ]

    loader_urls = UnstructuredURLLoader(urls=urls)
    documents_htmls = loader_urls.load()
```

This code loads text documents from different sources, such as text files, PDFs, and web pages.

1.  `text_loader_kwargs={'autodetect_encoding': True}`  - Set the  `autodetect_encoding`  option to  `True`  for the text loader.
2.  `loader_txt = DirectoryLoader("txt/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)`  - Initialize a  `DirectoryLoader`  object for loading text files from the "txt" directory.
    -   `DirectoryLoader`  - A loader for loading documents from a directory.
    -   `"txt/"`  - The directory path for the text files.
    -   `glob="./*.txt"`  - The glob pattern for matching text files.
    -   `loader_cls=TextLoader`  - The loader class for loading text files.
    -   `loader_kwargs=text_loader_kwargs`  - The loader arguments for the text loader.
3.  `documents_txt = loader_txt.load()`  - Load the text documents from the specified directory.
4.  The commented-out code block  `if 1==0:`  loads PDFs from the "pdfs" directory using  `PyPDFDirectoryLoader`.
5.  The second commented-out code block  `if 1==0:`  loads web pages from a list of URLs using  `UnstructuredURLLoader`.

After running this code, you will have the text documents loaded into memory as a list of  `Document`  objects. You can then use these documents for further processing, such as text classification, information extraction, or other natural language processing tasks.

**initialize embeddings:**

```
#################################################
##### Embeddings Model setup
##### Vectorization

text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)

all_splits = text_splitter.split_documents(documents_txt)

# specify embedding model (using huggingface sentence transformer)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
```

This code sets up the embedding model for vectorization of text documents.

1.  `text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)`  - Initialize a  `CharacterTextSplitter`  object for splitting documents into smaller chunks.
2.  `all_splits = text_splitter.split_documents(documents_txt)`  - Split the  `documents_txt`  list of  `Document`  objects into smaller chunks.
3.  `embedding_model_name = "sentence-transformers/all-mpnet-base-v2"`  - Specify the embedding model name using Hugging Face Sentence Transformer.
4.  `model_kwargs = {"device": "cuda"}`  - Set up the model arguments for the embedding model.
5.  `embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)`  - Initialize the  `HuggingFaceEmbeddings`  object for generating embeddings using the specified embedding model.

After running this code, you will have a  `HuggingFaceEmbeddings`  object that can be used for generating embeddings for the text chunks. These embeddings can then be used for various natural language processing tasks such as clustering, classification, or similarity search.

**initialize vectorstore:**

```

#document chunks and embiddings
vectordb = FAISS.from_documents(all_splits, embeddings)

retriever = vectordb.as_retriever()
```

This code creates a vector database using the FAISS library and a retriever for the document chunks and their corresponding embeddings.

1.  `vectordb = FAISS.from_documents(all_splits, embeddings)`  - Create a vector database using the FAISS library with the  `all_splits`  list of text chunks and their corresponding embeddings generated by the  `embeddings`  object.
2.  `retriever = vectordb.as_retriever()`  - Create a retriever object from the vector database for efficient similarity search.

After running this code, you will have a vector database and a retriever object that can be used for efficient similarity search and retrieval of document chunks based on their embeddings. 

**initialize chain**

```
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
```
This code creates a RetrievalQA chain using the specified language model (`llm`), retriever (`retriever`), and a prompt for the RetrievalQA chain.

1.  `RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})`  - Create a RetrievalQA chain using the specified language model (`llm`), retriever (`retriever`), and a prompt for the RetrievalQA chain.
    -   `llm`  - The language model to be used for generating answers.
    -   `retriever`  - The retriever object for efficient similarity search and retrieval of document chunks based on their embeddings.
    -   `chain_type_kwargs`  - A dictionary of keyword arguments for the RetrievalQA chain.
        -   `"prompt"`  - A prompt for the RetrievalQA chain.

After running this code, you will have a RetrievalQA chain that can be used for question answering tasks by combining the language model's ability to generate answers and the retriever's ability to efficiently search and retrieve relevant document chunks based on their embeddings.

## RetrievalQA Chain

We will first see how to do question answering after multiple relevant splits have been retrieved from the vector store. We may also need to compress the relevant splits to fit into the LLM context. Finally, we send these splits along with a system prompt and human question to the language model to get the answer.

![](https://miro.medium.com/v2/resize:fit:700/1*m1LdYJE0gl7zo_3Ifrzqdg.png)

Retrieval QA Chain

By default, we pass all the chunks into the same context window, into the same call of the language model. But, we can also use other methods in case the number of documents is high and if we can't pass them all in the same context window. MapReduce, Refine, and MapRerank are three methods that can be used if the number of documents is high. Now, we will look into these methods in detail.

**handle conversation**
```
# create conversation using rag in memory
def create_conversation(query: str, chat_history: list) -> tuple:
    try:
        start_time = time.time()
		result = qa_chain(query)
        chat_history.append((query, result["result"]))

#        return '', chat_history, text_to_speech(result['answer'])
        return '',chat_history, text_to_speech(result['result'])

    except Exception as e:
        chat_history.append((query, e))
        return '', chat_history, ''
```

This code defines a function  `create_conversation`  that takes a user query and a chat history as input and returns a tuple containing an empty string, the updated chat history, and a text-to-speech converted response.

1.  `def create_conversation(query: str, chat_history: list) -> tuple:`  - Define a function  `create_conversation`  that takes a user query (`query`) and a chat history (`chat_history`) as input and returns a tuple.
2.  `try:`  - Begin a try block for error handling.
3.  `start_time = time.time()`  - Record the start time for calculating the response time.
4.  `chat_history.append((query, result["result"]))`  - Append the user query and the corresponding response to the chat history.
5.  `return '',chat_history, text_to_speech(result['result'])`  - Return an empty string, the updated chat history, and a text-to-speech converted response.
6.  `except Exception as e:`  - Catch any exceptions that occur during the execution of the function.
7.  `chat_history.append((query, e))`  - Append the user query and the corresponding error message to the chat history.
8.  `return '', chat_history, ''`  - Return an empty string, the updated chat history, and an empty string.

The function  `create_conversation`  is designed to handle user queries and update the chat history with the corresponding responses or error messages. The text-to-speech converted response is also returned along with the chat history.

## RetrievalQA chain with Prompt

Let’s try to understand a little bit better what’s going on underneath the hood. First, we define the prompt template. The prompt template has instructions about how to use the context. It also has a placeholder for a context variable. We will use prompts to get answers to a question. Here, the prompt takes in the documents and the question and passes it to a language model.

**using gradio to speed up the chat bot building**

```

def bot(history):
    print("Question: ", history[-1][0])
    llm_chain.run(question=history[-1][0])
    history[-1][1] = ""
    for character in llm.streamer:
        print(character)
        history[-1][1] += character
        yield history

# build gradio ui
with gr.Blocks() as bot_interface:

    with gr.Row():
        chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column():
            html = gr.HTML()
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox()
    with gr.Row():
        with gr.Column():
            audio_input=gr.Audio(type="filepath")
            user_input = gr.Textbox()
            gr.Interface(
                fn=transcribe,
                inputs=[
                    audio_input
                ],
                outputs=[
                    user_input
                ],
                live=True)

#TEXT INPUT
    msg.submit(create_conversation, [msg, chatbot], [msg, chatbot, html])
```

This code defines a function  `bot`  that takes a chat history as input and generates a response using a language model and updates the chat history. It also builds a Gradio user interface for the chatbot.

1.  `def bot(history):`  - Define a function  `bot`  that takes a chat history as input.
2.  `print("Question: ", history[-1][0])`  - Print the user's question.
3.  `llm_chain.run(question=history[-1][0])`  - Run the language model with the user's question.
4.  `history[-1][1] = ""`  - Clear the previous response.
5.  `for character in llm.streamer:`  - Iterate over the characters in the language model's response.
6.  `history[-1][1] += character`  - Append each character to the response.
7.  `yield history`  - Yield the updated chat history.
8.  `with gr.Blocks() as bot_interface:`  - Define a Gradio user interface for the chatbot.
9.  `with gr.Row():`  - Define a row in the user interface.
10.  `chatbot = gr.Chatbot()`  - Define a chatbot component.
11.  `with gr.Row():`  - Define a row in the user interface.
12.  `with gr.Column():`  - Define a column in the user interface.
13.  `html = gr.HTML()`  - Define an HTML component.
14.  `with gr.Row():`  - Define a row in the user interface.
15.  `with gr.Column():`  - Define a column in the user interface.
16.  `msg = gr.Textbox()`  - Define a textbox for user input.
17.  `audio_input=gr.Audio(type="filepath")`  - Define an audio input component.
18.  `user_input = gr.Textbox()`  - Define a textbox for the transcription of the user's audio input.
19.  `gr.Interface(fn=transcribe, inputs=[audio_input], outputs=[user_input], live=True)`  - Define an interface for transcribing the user's audio input.
20.  `msg.submit(create_conversation, [msg, chatbot], [msg, chatbot, html])`  - Define a submit button for the textbox that triggers the  `create_conversation`  function.

The  `bot`  function generates a response using a language model and updates the chat history. The Gradio user interface allows the user to interact with the chatbot through text or audio input. The  `create_conversation`  function is called when the user submits a text input, updating the chat history with the user's question and the language model's response. The HTML component can be used to display additional information, such as the response time or the confidence score of the language model's response.

## limitations:
- This code uses the RetrievalQA chain, which is not the best option for dialogue and conversation; we used this chain due to the server's limited resources. (RetrievalQA is faster than other chains).
- As per the results, we will not be able to use the server in production.

## RetrievalQA limitations

One of the biggest disadvantages of RetrievalQA chain is that the QA chain fails to preserve conversational history. This can be checked as follows:
```
# Create a QA Chain  
qa_chain = RetrievalQA.from_chain_type(  
    llm,  
    retriever=vectordb.as_retriever()  
)
```
We will now ask a question to the chain.
```
question = "Is probability a class topic?"  
result = qa_chain({"query": question})  
result["result"]
```
Now, we will ask a second question to the chain.
```
question = "why are those prerequesites needed?"  
result = qa_chain({"query": question})  
result["result"]
```
We were able to get a reply from the chain which was not related to the previous answer. Basically, the RetrievalQA chain doesn’t have any concept of state. It doesn’t remember what previous questions or what previous answers were. We could In order for the chain to remember the previous question or previous answer, we need to introduce the concept of memory. This ability to remember the previous question or previous answer is required in the case of chatbots as we are able to ask follow-up questions to the chatbot or ask for clarification about previous answers.
