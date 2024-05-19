# import dependencies
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
#from langchain import hub


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

#from langchain import hub
#prompt = hub.pull("rlm/rag-prompt")

template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

prompt = PromptTemplate.from_template(template)

################################################
##### ASR
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
##### Large Language Model setup

model_name='mistralai/Mistral-7B-Instruct-v0.2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,config=model_config,device_map='auto')


# build huggingface pipeline for using Mistral-7B-Instruct
#streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#streamer  = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

pipeline = pipeline(task='text-generation',
        model=model,
        tokenizer=tokenizer,
#        temperature=0.1,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1500,
        do_sample=False,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id,
#        streamer = streamer
)

llm = HuggingFacePipeline(pipeline=pipeline)

#################################################
##### Loading data-sources
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
        "https://www.ultatel.com/cloud-based-phone-system/",
        "https://www.ultatel.com/cloud-contact-center/",
        "https://www.ultatel.com/microsoft-teams-contact-center/",
        "https://www.ultatel.com/hub-app/",
        "https://www.ultatel.com/voip-business-phone/",
        "https://www.ultatel.com/plans-prices/"
    ]

    loader_urls = UnstructuredURLLoader(urls=urls)
    documents_htmls = loader_urls.load()

#################################################
##### Embeddings Model setup
##### Vectorization

# splitting doc
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=65, chunk_overlap=0) #Chage the chunk_size and chunk_overlap as needed
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)


#all_splits_pdfs = text_splitter.split_documents(documents_pdfs)
#all_splits_htmls = text_splitter.split_documents(documents_htmls)
#all_splits = all_splits_pdfs + all_splits_htmls
# print (all_splits_pdfs)
# print (all_splits_htmls)
# print (all_splits)

all_splits = text_splitter.split_documents(documents_txt)

# specify embedding model (using huggingface sentence transformer)
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)


#document chunks and embiddings
vectordb = FAISS.from_documents(all_splits, embeddings)

retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

################################################
##### RAG + TTS
# create conversation using rag in memory
def create_conversation(query: str, chat_history: list) -> tuple:
    try:
        start_time = time.time()
#        print("--- %s seconds ---" % (time.time() - start_time))

###        memory = ConversationBufferMemory(
###            memory_key='chat_history',
###            return_messages=False
###        )

#        memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=False)
#        print("---initiate Conversation-memory: %s seconds ---" % (time.time() - start_time))
#        start_time = time.time()

#        qa_chain = ConversationalRetrievalChain.from_llm(
#            llm=llm,
#            retriever=retriever,
#            memory=memory,
#            get_chat_history=lambda h: h,
#        )
#        print("---initiate ConversationalRetrievalChain: %s seconds ---" % (time.time() - start_time))
#        start_time = time.time()

#        result = qa_chain({'question': query, 'chat_history': chat_history})

        result = qa_chain.invoke(query)

        print("---answering query: %s seconds ---" % (time.time() - start_time))

#        start_time = time.time()

#        chat_history.append((query, result['answer']))
        chat_history.append((query, result["result"]))

#        return '', chat_history, text_to_speech(result['answer'])
        return '',chat_history, text_to_speech(result['result'])

    except Exception as e:
        chat_history.append((query, e))
        return '', chat_history, ''
################################################

def text_to_speech(text:str) -> str:

    start_time = time.time()

    tts = gTTS(text)
    tts.save('output.mp3')

    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = f'<audio hidden src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

    print("---TTS generation: %s seconds ---" % (time.time() - start_time))

    return audio_player



################################################
##### UI + Orechestration

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


#ASR INPUT
#    inputs_event = audio_input.stop_recording(transcribe, audio_input, user_input)
#    inputs_event.then(create_conversation, [user_input, chatbot], [user_input, chatbot, html])
    inputs_event = audio_input.stop_recording(create_conversation, [user_input, chatbot], [user_input, chatbot, html])

bot_interface.title = 'Chat with Mistral 7B'
bot_interface.launch(debug=False,share=True)
