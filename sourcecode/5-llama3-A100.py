import time
import torch
from IPython.display import display_markdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
import transformers
from langchain.document_loaders import UnstructuredPDFLoader,PDFMinerLoader,TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# Hugging Face model id
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,config=model_config,device_map='auto')

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer = tokenizer,
    model=model,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
    },
)

terminators =  [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


class Llama3_8B_gen:
    def __init__(self,pipeline):
        self.pipeline= pipeline

    @staticmethod
    def generate_prompt(query,retrieved_text):
        messages = [
            {"role": "system", "content": "Answer the Question for the Given below context and information and not prior knowledge, only give the output result \n\ncontext:\n\n{}".format(retrieved_text) },
            {"role": "user", "content": query},]
        return pipeline.tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)

    def generate(self,query,retrieved_context):
        prompt = self.generate_prompt(query ,retrieved_context)
       output =  pipeline(prompt,max_new_tokens=512,eos_token_id=terminators,do_sample=False,)
        return output[0]["generated_text"][len(prompt):]

class Langchain_RAG:
    def __init__(self):
#        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        text_loader_kwargs={'autodetect_encoding': True}
        loader_txt = DirectoryLoader("txt/", glob="./*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        documents_txt = loader_txt.load()
        text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)


        self.texts = text_splitter.split_documents(documents_txt)
        self.get_vec_value= FAISS.from_documents(self.texts, self.embeddings)
        self.retriever = self.get_vec_value.as_retriever(search_kwargs={"k": 4})

    def __call__(self,query):
        rev = self.retriever.get_relevant_documents(query)
        return "".join([i.page_content for i in rev])


text_gen = Llama3_8B_gen(pipeline=pipeline)
retriever = Langchain_RAG()


def Rag_qa(query):
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate(query,retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "what is blacklist feature?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("what is blacklist feature?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "what is voicemail?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("what is voicemail?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "how can I activate voicemail?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("how can I activate voicemail?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "Is there any Cisco device cheaper than 160$?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("Is there any Cisco device cheaper than 160$?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "What is the price of Grandstream GXV3275?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("What is the price of Grandstream GXV3275?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "Is Grandstream GXV3275 better than Yealink EXP50?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("Is Grandstream GXV3275 better than Yealink EXP50?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "Is Grandstream GXV3275 Cheaper than Yealink EXP50?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("Is Grandstream GXV3275 Cheaper than Yealink EXP50?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))

    query = "Is there any visual represntation on the phone for a new voicemail?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("Is there any visual represntation on the phone for a new voicemail?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))
    return result


print (Rag_qa("Cisco 7841"))


