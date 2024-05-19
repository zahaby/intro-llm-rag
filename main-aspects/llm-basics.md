
## What is a large language model (LLM)?

A large language model (LLM) is a type of  [artificial intelligence (AI)](https://www.cloudflare.com/learning/ai/what-is-artificial-intelligence/)  program that can recognize and generate text, among other tasks. LLMs are trained on  [huge sets of data](https://www.cloudflare.com/learning/ai/big-data/)  — hence the name "large." LLMs are built on  [machine learning](https://www.cloudflare.com/learning/ai/what-is-machine-learning/): specifically, a type of  [neural network](https://www.cloudflare.com/learning/ai/what-is-neural-network/)  called a transformer model.

In simpler terms, an LLM is a computer program that has been fed enough examples to be able to recognize and interpret human language or other types of complex data. Many LLMs are trained on data that has been gathered from the Internet — thousands or millions of gigabytes' worth of text. But the quality of the samples impacts how well LLMs will learn natural language, so an LLM's programmers may use a more curated data set.

LLMs use a type of machine learning called  [deep learning](https://www.cloudflare.com/learning/ai/what-is-deep-learning/)  in order to understand how characters, words, and sentences function together. Deep learning involves the probabilistic analysis of unstructured data, which eventually enables the deep learning model to recognize distinctions between pieces of content without human intervention.

LLMs are then further trained via tuning: they are fine-tuned or prompt-tuned to the particular task that the programmer wants them to do, such as interpreting questions and generating responses, or translating text from one language to another.

## How do LLMs work?

#### Machine learning and deep learning

At a basic level, LLMs are built on machine learning. Machine learning is a subset of AI, and it refers to the practice of feeding a program large amounts of data in order to train the program how to identify features of that data without human intervention.

LLMs use a type of machine learning called deep learning. Deep learning models can essentially train themselves to recognize distinctions without human intervention, although some human fine-tuning is typically necessary.

Deep learning uses probability in order to "learn." For instance, in the sentence "The quick brown fox jumped over the lazy dog," the letters "e" and "o" are the most common, appearing four times each. From this, a deep learning model could conclude (correctly) that these characters are among the most likely to appear in English-language text.

Realistically, a deep learning model cannot actually conclude anything from a single sentence. But after analyzing trillions of sentences, it could learn enough to predict how to logically finish an incomplete sentence, or even generate its own sentences.

#### Neural networks

In order to enable this type of deep learning, LLMs are built on neural networks. Just as the human brain is constructed of neurons that connect and send signals to each other, an artificial neural network (typically shortened to "neural network") is constructed of network nodes that connect with each other. They are composed of several "layers”: an input layer, an output layer, and one or more layers in between. The layers only pass information to each other if their own outputs cross a certain threshold.

#### Transformer models

The specific kind of neural networks used for LLMs are called transformer models. Transformer models are able to learn context — especially important for human language, which is highly context-dependent. Transformer models use a mathematical technique called self-attention to detect subtle ways that elements in a sequence relate to each other. This makes them better at understanding context than other types of machine learning. It enables them to understand, for instance, how the end of a sentence connects to the beginning, and how the sentences in a paragraph relate to each other.

This enables LLMs to interpret human language, even when that language is vague or poorly defined, arranged in combinations they have not encountered before, or contextualized in new ways. On some level they "understand" semantics in that they can associate words and concepts by their meaning, having seen them grouped together in that way millions or billions of times.

ref:https://www.cloudflare.com/learning/ai/what-is-large-language-model/

## What are the Relations and Differences between LLMs and Transformers?
**Transformers**

Has gained a lot of popularity in the field of natural language processing (NLP). These are good at understanding the relationships between words in a sentence or sequence of text. Unlike traditional models like RNNs, Transformers don't rely on sequential processing, allowing them to do computation in parallel and process sentences more efficiently. Overall, these are powerful models good at understanding relationships between words and have modernized NLP area.

Imagine a sentence: "The cat sat on the mat." A transformer breaks down this sentence into smaller units called "tokens" (e.g., "The," "cat," "sat," "on," "the," "mat," and punctuation marks). Each token is represented as a vector, capturing its meaning and context. The transformer then learns to analyse the relationships between these tokens to understand the sentence's overall meaning.

Example models,

-   BERT (Bidirectional Encoder Representations from Transformers)
-   GPT (Generative Pre-trained Transformer)
-   T5 (Text-to-Text Transfer Transformer)
-   DialoGPT

**LLM (Large Language Model)**

Is a specific type of transformer that has been trained on vast amounts of text data. It has learned to predict the next word in a sentence given the context of the previous words. This ability allows LLMs to generate contextually correct text.

For instance, if you provide the prompt "Once upon a time in a land far" an LLM can generate the next words as "away." The LLM bases its predictions on the patterns and context it has learned during training on massive amounts of text. This makes LLMs useful for various applications, such as auto-completion, translation, summarization, and even creative writing.

-   GPT3.5 Turbo & GPT-4 by OpenAI
-   BLOOM by BigScience
-   LaMDA by Google
-   MT-NLG by Nvidia/Microsoft2
-   LLaMA by Meta AI2

**Relation and Differences between LLMs and Transformers**

Transformers and LLMs (Large Language Models) are related concepts, as LLMs are a specific type of model that is built upon the transformer architecture. While transformers, in general, can be used for various tasks beyond language modeling, LLMs are specifically trained in generating text and understanding natural language(There can be exceptions as this field is quickly evolving, and pace of research and funding is unprecedented).

The main differences between transformers and LLMs lie in their specific purposes and training objectives. Transformers are a broader class of models that can be applied to various tasks, including language translation, speech recognition, and image captioning, while LLMs are focused on language modeling and text generation(there are some exceptions). Transformers serve as the underlying architecture that enables LLMs to understand and generate text by capturing contextual relationships and long-range dependencies. Transformers are more general-purpose models, whereas LLMs are specifically trained and optimized for language modeling and generation tasks.

Transformer models can also be divided into three categories: encoders, decoders, and encoder-decoder architectures. This categorization is based on the different roles these components play in the model's overall function.  **Encoders**  aim to understand the input sequence. They focus on processing the input and capturing its meaning and context.  **Decoders**, on the other hand, generate output based on the information learned by the encoder. They take the encoded representations and produce the desired output sequence.  **Encoder-decoder**  models combine both encoder and decoder components. They are used in tasks where the input and output sequences have different lengths or meanings. The encoder understands the input sequence, and the decoder generates the corresponding output sequence.

ref:https://www.linkedin.com/pulse/transformers-llms-next-frontier-ai-vijay-chaudhary/

## What are Pipelines in Transformers?
-   They provide an easy-to-use API through pipeline() method for performing inference over a variety of tasks.
-   They are used to encapsulate the overall process of every Natural Language Processing task, such as text cleaning, tokenization, embedding, etc.

The pipeline() method has the following structure:

```
from transformers import pipeline

# To use a default model & tokenizer for a given task(e.g. question-answering)
pipeline("task-name")

# To use an existing model
pipeline("task-name", model="model_name")

# To use a custom model/tokenizer
pipeline('task-name', model='model name',tokenizer='tokenizer_name')
```
>This code snippet is using the transformers library to create a pipeline for natural language processing tasks such as question-answering.
- The first line imports the pipeline function from the transformers library.
- The next three lines show how to use the pipeline function for different scenarios.
- The first scenario uses a default model and tokenizer for a given task, which is specified in the placeholder "task-name".
- The second scenario uses an existing model, which is specified in the placeholder "model_name", for the same task as in the first scenario.
- The third scenario uses a custom model and tokenizer, which are specified in the placeholders "model name" and "tokenizer_name", respectively, for the same task as in the first two scenarios.
- Overall, the pipeline function allows for easy implementation of natural language processing tasks with various models and tokenizers.

ref:https://www.datacamp.com/tutorial/an-introduction-to-using-transformers-and-hugging-face

## What are Hugging Face Transformers?

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index) is an open-source framework for deep learning created by Hugging Face. It provides APIs and tools to download state-of-the-art pre-trained models and further tune them to maximize performance. These models support common tasks in different modalities, such as natural language processing, computer vision, audio, and multi-modal applications.
For many applications, such as sentiment analysis and text summarization, pre-trained models work well without any additional model training.

Hugging Face Transformers pipelines encode best practices and have default models selected for different tasks, making it easy to get started. Pipelines make it easy to use GPUs when available and allow batching of items sent to the GPU for better throughput performance.

Hugging Face provides:

-   A  [model hub](https://huggingface.co/models)  containing many pre-trained models.
    
-   The  [🤗 Transformers library](https://huggingface.co/docs/transformers/index)  that supports the download and use of these models for NLP applications and fine-tuning. It is common to need both a tokenizer and a model for natural language processing tasks.
    
-   [🤗 Transformers pipelines](https://huggingface.co/docs/transformers/v4.26.1/en/pipeline_tutorial)  that have a simple interface for most natural language processing tasks.

ref:https://docs.databricks.com/en/machine-learning/train-model/huggingface/index.html
