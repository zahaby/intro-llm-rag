
## The Curse Of The LLMs

As usage exploded, so did the expectations. Many users started using ChatGPT as a source of information, like an alternative to Google. As a result, they also started encountering prominent weaknesses of the system. Concerns around copyright, privacy, security, ability to do mathematical calculations etc. aside, people realised that there are two major limitations of Large Language Models.

![](https://media.licdn.com/dms/image/D5612AQHbj26__iauNA/article-inline_image-shrink_1500_2232/0/1701952601021?e=1719446400&v=beta&t=8WDKYTmbylOm7ct5RhuuCz5AvIp4sR5xp8Q4NM8vFuk)

Curse of the LLMs

> _Users look at LLMs for knowledge and wisdom, yet LLMs are sophisticated predictors of what word comes next._
## The Challenge

-   Make LLMs respond with up-to-date information
-   Make LLMs not respond with factually inaccurate information
-   Make LLMs aware of proprietary information

## What is RAG?

In 2023, RAG has become one of the most used technique in the domain of Large Language Models.

![](https://media.licdn.com/dms/image/D5612AQFiRxJdS1arSA/article-inline_image-shrink_1500_2232/0/1701952773074?e=1719446400&v=beta&t=Ct043QYFp4cq8xVzRcCvq2Vrm_gPzMrt01Z8RL9L0hs)

Retrieval Augmented Generation

_User writes a prompt or a query that is passed to an orchestrator_

_Orchestrator sends a search query to the retriever_

_Retriever fetches the relevant information from the knowledge sources and sends back_

_Orchestrator augments the prompt with the context and sends to the LLM_

_LLM responds with the generated text which is displayed to the user via the orchestrator_

## How does RAG help?

### Unlimited Knowledge

The Retriever of an RAG system can have access to external sources of information. Therefore, the LLM is not limited to its internal knowledge. The external sources can be proprietary documents and data or even the internet.

![](https://media.licdn.com/dms/image/D5612AQE2pMAHS73egQ/article-inline_image-shrink_1500_2232/0/1701952654042?e=1719446400&v=beta&t=nGCqksbSvchyQmW9IRjemMZNzirfC9vSGEINY3w6gtw)

Expanding LLM Memory with RAG

### Confidence in Responses

With the context (extra information that is retrieved) made available to the LLM, the confidence in LLM responses is increased.

![](https://media.licdn.com/dms/image/D5612AQEiaJnX8bEWDg/article-inline_image-shrink_1500_2232/0/1701952653609?e=1719446400&v=beta&t=1scuNje5y0PLgTshXdXfTIVFpa0BdwBwGxfD1gddWLo)

Increasing Confidence in LLM Responses

As RAG technique evolves and becomes accessible with frameworks like  [LangChain](https://www.linkedin.com/company/langchain/)  and  [LlamaIndex](https://www.linkedin.com/company/llamaindex/)  , it is finding more and more application in LLM powered applications like QnA with documents, conversational agents, recommendation systems and for content generation.

ref:https://www.linkedin.com/pulse/context-key-significance-rag-language-models-abhinav-kimothi-nebnc/

𝗡𝗘𝗪 𝗥𝗔𝗚 𝘁𝗲𝗰𝗵𝗻𝗶𝗾𝘂𝗲𝘀 :-  
  
1. 𝗖𝗵𝗮𝗶𝗻 𝗼𝗳 𝗡𝗼𝘁𝗲 - Steps in CoN involve Generating notes for documents that have been retrieved, which result in a more factually correct answer and also because Notes are generated at steps that have been used to break the problem in the final step trustworthiness of the answer also increases.  
  https://cobusgreyling.medium.com/chain-of-note-con-retrieval-for-llms-763ead1ae5c5
  
2. 𝗖𝗼𝗿𝗿𝗲𝗰𝘁𝗶𝘃𝗲 𝗥𝗔𝗚 = This RAG technique breaks the problem into a binary step if the retrieved answer is Ambiguous --> Then the query is passed to Search and then search results are taken and finally LLM is triggered again to look at the query keeping in mind both RAG document and Search results.  
  https://medium.com/the-ai-forum/implementing-a-flavor-of-corrective-rag-using-langchain-chromadb-zephyr-7b-beta-and-openai-30d63e222563
  
3. 𝗥𝗔𝗚 𝗙𝘂𝘀𝗶𝗼𝗻 - A Query is broken into small sub-queries in this approach. Then these queries are given to a vector DB to retrieve the most relevant documents for each query. Finally, using the Reciprocal rank fusion algorithm, the most relevant information is prioritized.  
( In  [LlamaIndex](https://www.linkedin.com/company/llamaindex/)  When I used the combination of Recursive Retrieval and Semantic Chunking +  [Pinecone](https://www.linkedin.com/company/pinecone-io/)  as VectorDB results came out best for our RAG application)  

- RAG-Fusion improves traditional search systems by overcoming their limitations
through a multi-query approach. It expands user queries into multiple diverse
perspectives using a Language Model (LLM). This strategy goes beyond capturing
explicit information and delves into uncovering deeper, transformative
knowledge. The fusion process involves conducting parallel vector searches for
both the original and expanded queries, intelligently re-ranking to optimize
results, and pairing the best outcomes with new queries.
https://medium.com/@kbdhunga/advanced-rag-rag-fusion-using-langchain-772733da00b7
  
  
4. 𝗦𝗲𝗹𝗳-𝗥𝗔𝗚 - A self-rag technique where LLMs can do self-reflection for dynamic retrieval, critique, and generation.  
 https://github.com/run-llama/llama-hub/blob/main/llama_hub/llama_packs/self_rag/self_rag.ipynb
 https://cobusgreyling.medium.com/self-reflective-retrieval-augmented-generation-self-rag-f5cbad4412d5

ref: https://www.linkedin.com/feed/update/urn:li:activity:7185147270554681344/?commentUrn=urn%3Ali%3Acomment%3A(ugcPost%3A7183502198595629058%2C7186165836728979456)&dashCommentUrn=urn%3Ali%3Afsd_comment%3A(7186165836728979456%2Curn%3Ali%3AugcPost%3A7183502198595629058)&dashReplyUrn=urn%3Ali%3Afsd_comment%3A(7186182955751354368%2Curn%3Ali%3AugcPost%3A7183502198595629058)&replyUrn=urn%3Ali%3Acomment%3A(ugcPost%3A7183502198595629058%2C7186182955751354368)

https://www.linkedin.com/feed/update/urn:li:activity:7180436217006600194/
