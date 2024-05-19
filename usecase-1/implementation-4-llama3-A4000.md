
- We use 2XA4000 GPUs with low memory and the llama3 model in this experiment.

# Code Implementation

- You have to grand access by email first here: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Then use huggingface access-token (https://huggingface.co/settings/tokens) to login : https://huggingface.co/welcome `huggingface-cli login`
- Use the model
```
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
```
- Setup quantization (to be able to use the model with low resource)
```
device_ids = [0, 1]  # Modify this list according to your GPU configuration
primary_device = f'cuda:{device_ids[0]}'  # Primary device
torch.cuda.set_device(primary_device)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=config,config=model_config,device_map='auto')

model = torch.nn.DataParallel(model, device_ids=device_ids)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

pipeline = transformers.pipeline(
    "text-generation",
    tokenizer = tokenizer,
    model=model.module,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
    },
)
```

- This model using a different terminator
```
terminators =  [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
```
- Setup prompt (prompt here has a different config)
```
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
```

- Setup RAG
```
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
```
- Finally query the model+rag
```
    query = "what is blacklist feature?"
    start_time = time.time()
    retriever_context = retriever(query)
    result = text_gen.generate("what is blacklist feature?",retriever_context)
    print(result)
    print("--- answering in: %s seconds ---" % (time.time() - start_time))
```
