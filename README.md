# LLM Models and RAG Hands-on Guide

Welcome to the LLM Models and RAG Hands-on Guide repository! This guide is designed for technical teams interested in developing basic conversational AI solutions using Retrieval-Augmented Generation (RAG).

## Introduction

This repository provides a comprehensive guide for building conversational AI systems using large language models (LLMs) and RAG techniques. The content combines theoretical knowledge with practical code implementations, making it suitable for those with a basic technical background.



## Table of Contents

This guide is primarily for technical teams engaged in developing a basic conversational AI with RAG solutions. It offers a basic introduction to the technical aspects.
This guide helps anyone with basic technical background to get involved in the AI domain.
This guide combines between the theoretical, basic knowledge and code implementation.
It's important to note that most of the content is compiled from various online resources, reflecting the extensive effort in
curating and organizing this information from numerous sources.

- [intro](https://github.com/zahaby/intro-llm-rag/blob/main/intro.md)
  - [What is Conversational AI?](https://github.com/zahaby/intro-llm-rag/blob/main/intro.md#what-is-conversational-ai)
  - [The Technology Behind Conversational AI](https://github.com/zahaby/intro-llm-rag/blob/main/intro.md#the-technology-behind-conversational-ai)
  - [LLM Basics](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md)
  - [What is a large language model (LLM)? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md#what-is-a-large-language-model-llm)
  - [How do LLMs work? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md#how-do-llms-work)
  - [What are the Relations and Differences between LLMs and Transformers?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md#what-are-the-relations-and-differences-between-llms-and-transformers)
  - [What are Pipelines in Transformers? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md#what-are-pipelines-in-transformers)
  - [What are Hugging Face Transformers?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/llm-basics.md#what-are-pipelines-in-transformers)
  - [Chains](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md)
  - [What are chains?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#what-are-chains)
  - [Foundational chain types in LangChain ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#foundational-chain-types-in-langchain)
  - [LLMChain ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#llmchain)
  - [Creating an LLMChain](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#creating-an-llmchain)
  - [Sequential Chains ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#equential-chains)
  - [SimpleSequentialChain](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#simplesequentialchain)
  - [SequentialChain](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#sequentialchain)
  - [Transformation ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chains.md#transformation)
  - [Prompt Engineering](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md)
  - [What is Prompt Engineering? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#what-is-prompt-engineering)
    - [Prompt](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#prompt)
    - [Types of Prompts ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#types-of-prompts)
    - [Instruction Prompting ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#instruction-prompting)
    - [Role Prompting ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#role-prompting)
    - [“Standard” Prompting ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#standard-prompting)
    - [Chain of Thought (CoT) Prompting](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#chain-of-thought-cot-prompting)
    - [Recommendations and Tips for Prompt Engineering with OpenAI API](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/prompt-engineering.md#recommendations-and-tips-for-prompt-engineering-with-openai-api)
  - [Embeddings](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md)
    - [A problem with semantic search.](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md#a-problem-with-semantic-search)
    - [What are embeddings? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md#what-are-embeddings)
    - [What is a vector in machine learning?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md#what-is-a-vector-in-machine-learning)
    - [How do embeddings work?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md#how-do-embeddings-work)
    - [How are embeddings used in large language models (LLMs)?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/embeddings.md#how-are-embeddings-used-in-large-language-models-llms)
  - [Vector Stores](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md)
    - [What Are Vector Databases? ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#what-are-vector-databases)
    - [The Benefits of Using Open Source Vector Databases](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#the-benefits-of-using-open-source-vector-databases)
    - [Open Source Vector Databases Comparison: Chroma Vs. Milvus Vs. Weaviate](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#open-source-vector-databases-comparison-chroma-vs-milvus-vs-weaviate)
      - [1. Chroma ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#1-chroma)
      - [2. Milvus](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#2-milvus)
      - [3. Weaviate](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#3-weaviate)
      - [4.Faiss](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/vectorstores.md#4faiss)
  - [Chunking](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md)
    - [Document Splitting](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#document-splitting)
    - [Chunking Methods ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#chunking-strategies)
    - [Character Splitting ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#character-splitting)
    - [Recursive Character Text Splitting ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#recursive-character-text-splitting)
    - [Split by Tokens](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#split-by-tokens)
      - [Tiktoken Tokenizer ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#tiktoken-tokenizer)
      - [Hugging Face Tokenizer](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#hugging-face-tokenizer)
      - [Other Tokenizer](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#other-tokenizer)
    - [Things to Keep in Mind](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/chunking.md#things-to-keep-in-mind)
  - [Quantization ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md)
    - [What is Quantization?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#what-is-quantization)
    - [How does quantization work?](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#how-does-quantization-work)
    - [Hugging Face and Bitsandbytes Uses](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#hugging-face-and-bitsandbytes-uses)
    - [Loading a Model in 4-bit Quantization](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#loading-a-model-in-4-bit-quantization)
    - [Loading a Model in 8-bit Quantization](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#loading-a-model-in-8-bit-quantization)
    - [Changing the Compute Data Type ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#changing-the-compute-data-type)
    - [Using NF4 Data Type ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#using-nf4-data-type)
    - [Nested Quantization for Memory Efficiency ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#nested-quantization-for-memory-efficiency)
    - [Loading a Quantized Model from the Hub ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#loading-a-quantized-model-from-the-hub)
    - [Exploring Advanced techniques and configuration](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/quantization.md#exploring-advanced-techniques-and-configuration)
  - [Temperature   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md)
    - [Top P and Temperature   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#top-p-and-temperature)
    - [Temperature   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#temperature)
    - [Top p  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#top-p)
    - [Token length  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#token-length)
    - [Max tokens  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#max-tokens)
    - [Stop tokens  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/temprature.md#stop-tokens)
  - [Langchain Memory  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md)
    - [What is Conversational memory?  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#what-is-conversational-memory)
    - [ConversationChain  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#conversationchain)
    - [Forms of Conversational Memory   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#forms-of-conversational-memory)
      - [ConversationBufferMemory  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#conversationbuffermemory)
      - [ConversationSummaryMemory   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#conversationsummarymemory)
      - [ConversationBufferWindowMemory  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#conversationbufferwindowmemory)
      - [ConversationSummaryBufferMemory  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#conversationsummarybuffermemory)
      - [Other Memory Types   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/langchain-memory.md#other-memory-types)
  - [Agents & Tools  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md)
    - [Tools  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#tools)
    - [Agents  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#agents)
    - [Chains  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#chains)
    - [Memory   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#memory)
    - [Callback Handlers ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#callback-handlers)
  - [Walkthrough — Project Utilizing Langchain  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/agents-tools.md#walkthrough--project-utilizing-langchain)
  - [RAG  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md)
    - [The Curse Of The LLMs   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md#the-curse-of-the-llms)
    - [The Challenge ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md#the-challenge)
    - [What is RAG?  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md#what-is-rag)
    - [How does RAG help?  ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md#how-does-rag-help)
    - [New RAG techniques   ](https://github.com/zahaby/intro-llm-rag/blob/main/main-aspects/rag.md#%F0%9D%97%A1%F0%9D%97%98%F0%9D%97%AA-%F0%9D%97%A5%F0%9D%97%94%F0%9D%97%9A-%F0%9D%98%81%F0%9D%97%B2%F0%9D%97%B0%F0%9D%97%B5%F0%9D%97%BB%F0%9D%97%B6%F0%9D%97%BE%F0%9D%98%82%F0%9D%97%B2%F0%9D%98%80--)
- [groq  ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/groq.md)
  - [What is groq?   ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/groq.md)
  - [What is LPU?  ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/lpu.md#what-is-lpu)
  - [How Groq's LPU Works  ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/lpu.md#how-groqs-lpu-works)
  - [How LPU is different from GPU   ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/lpu.md#how-lpu-is-different-from-gpu)
  - [Groq Tools   ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/groq-tools.md)
  - [Groq and RAG Architecture Example   ](https://github.com/zahaby/intro-llm-rag/blob/main/groq/groq-rag.md)
- [What is LlamaParse ?  ](https://github.com/zahaby/intro-llm-rag/blob/main/llama-parsing/llama-parser.md)
- [Use Case – 1   ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/implementation-1-A4000.md)
  - [implementation-1-A4000   ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/implementation-1-A4000.md#code-implementation)
  - [implementation-2-A100   ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/implementation-2-A100.md)
  - [implementation-3-groq  ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/implementation-3-groq.md)
  - [implementation-4-llama3-A4000  ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/implementation-4-llama3-A4000.md)
  - [benchmark](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-1/benchmark.md)
- [Use Case – 2   ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-2/google-calendar.md)
  - [Action integration with chatbot (google calendar booking)  ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-2/google-calendar.md#code-implementation)
  - [Action integration with chatbot (get weather)  ](https://github.com/zahaby/intro-llm-rag/blob/main/usecase-2/get_weather.md)
- [Source Code  ](https://github.com/zahaby/intro-llm-rag/tree/main/sourcecode)


## Key Concepts

### Conversational AI

An introduction to the technology behind conversational AI, covering its fundamentals and applications.

### Large Language Models (LLMs)

Understand what LLMs are, how they work, and their role in conversational AI. This section also explores the differences between LLMs and transformers.

### Transformers

Detailed explanation of transformers, including their pipelines and the Hugging Face library.

### Prompt Engineering

Learn about different types of prompts, prompt engineering techniques, and best practices for using the OpenAI API.

### Embeddings and Vector Stores

Explore the use of embeddings in LLMs, vector databases, and various chunking methods for document splitting.

## Hands-on Examples

### Use Case 1

Implementation details for the first use case, including benchmark results and performance analysis. Refer to the `usecase-1` directory for code and documentation.

### Use Case 2

A detailed walkthrough of integrating actions with a chatbot, such as getting weather event. See the `usecase-2` directory for more information.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Contributors

Please feel free to contribute to enrich the content! 

## Contact

For any questions or feedback, please feel free to contact me directly @[zahaby](https://www.linkedin.com/in/zahaby/).

---

Happy coding!
