
### A problem with semantic search

The basic design of a semantic search system, as pitched by most vector search vendors, has two  _easy_ (this is irony)  steps:

1.  Compute embeddings for your documents and queries. Somewhere. Somehow. Figure it out by yourself.
2.  Upload them to a vector search engine and enjoy a better semantic search.

![](https://miro.medium.com/v2/resize:fit:700/1*CPNviIUwdayBj56LGs0KPA.png)

A good embedding model is essential for semantic search. Image by author.

**Your semantic search is as good as your embedding model**, but choosing the model is often considered out of scope for most early adopters. So everyone just takes a  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  and hopes for the best.

But this approach has more open questions than answers:

-   Is there a difference between embedding models? Are paid models from OpenAI and Cohere better?
-   How do they handle multiple languages? Is there a benefit in large 1B+ models?
-   Dense retrieval using embeddings is one of many semantic search methods. Is it better than new-age sparse approaches like  [SPLADEv2](https://arxiv.org/abs/2109.10086)  and  [ELSER](https://www.elastic.co/guide/en/machine-learning/8.8/ml-nlp-elser.html)?

## What are embeddings?

Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by  [machine learning](https://www.cloudflare.com/learning/ai/what-is-machine-learning/)  models and semantic search algorithms. They translate objects like these into a mathematical form according to the factors or traits each one may or may not have, and the categories they belong to.

Essentially, embeddings enable machine learning models to find similar objects. Given a photo or a document, a machine learning model that uses embeddings could find a similar photo or document. Since embeddings make it possible for computers to understand the relationships between words and other objects, they are foundational for  [artificial intelligence (AI)](https://www.cloudflare.com/learning/ai/what-is-artificial-intelligence/).

For example, the documents in the upper right of this two-dimensional space may be relevant to each other:

![Embeddings - Documents in vector space clustered together](https://cf-assets.www.cloudflare.com/slt3lc6tev37/6GPsu7uHy0hGNfHXbfQvis/bf3c0b03654368a2783168ea76858326/vector_database_clusters.png "What are embeddings? - example")

Technically, embeddings are  _vectors_  created by machine learning models for the purpose of capturing meaningful data about each object.

## What is a vector in machine learning?

In mathematics, a vector is an array of numbers that define a point in a dimensional space. In more practical terms, a vector is a list of numbers — like 1989, 22, 9, 180. Each number indicates where the object is along a specified dimension.

In machine learning, the use of vectors makes it possible to search for similar objects. A vector-searching algorithm simply has to find two vectors that are close together in a  [vector database](https://www.cloudflare.com/learning/ai/what-is-vector-database/).

To understand this better, think about latitude and longitude. These two dimensions — north-south and east-west, respectively — can indicate the location of any place on Earth. The city of Vancouver, British Columbia, Canada can be represented as the latitude and longitude coordinates 49°15'40"N, 123°06'50"W. This list of two values is a simple vector.

Now, imagine trying to find a city that is very near Vancouver. A person would just look at a map, while a machine learning model could instead look at the latitude and longitude (or vector) and find a place with a similar latitude and longitude. The city of Burnaby is at 49°16'N, 122°58'W — very close to 49°15'40"N, 123°06'50"W. Therefore, the model can conclude, correctly, that Burnaby is located near Vancouver.

#### Adding more dimensions to vectors

Now, imagine trying to find a city that is not only close to Vancouver, but of similar size. To this model of locations, let us add a third "dimension" to latitude and longitude: population size. Population can be added to each city's vector, and population size can be treated like a Z-axis, with latitude and longitude as the Y- and X-axes.

The vector for Vancouver is now 49°15'40"N, 123°06'50"W, 662,248*. With this third dimension added, Burnaby is no longer particularly close to Vancouver, as its population is only 249,125*. The model might instead find the city of Seattle, Washington, US, which has a vector of 47°36'35"N 122°19'59"W, 749,256**.

_*As of 2021.  
**As of 2022._

This is a fairly simple example of how vectors and similarity search work. But to be of use, machine learning models may want to generate more than three dimensions, resulting in much more complex vectors.

#### Even more multi-dimensional vectors

For instance, how can a model tell which TV shows are similar to each other, and therefore likely to be watched by the same people? There are any number of factors to take into account: episode length, number of episodes, genre classification, number of viewers in common, actors in each show, year each show debuted, and so on. All of these can be "dimensions," and each show represented as a point along each of these dimensions.

Multi-dimensional vectors can help us determine if the sitcom  _Seinfeld_  is similar to the horror show  _Wednesday_.  _Seinfeld_  debuted in 1989,  _Wednesday_  in 2022. The two shows have different episode lengths, with  _Seinfeld_  at 22-24 minutes and  _Wednesday_  at 46-57 minutes — and so on. By looking at their vectors, we can see that these shows likely occupy very different points in a dimensional representation of TV shows.

TV show,  Genre,  Year debuted,  Episode length, Seasons (through 2023), Episodes (through 2023)

We can express these as vectors, just as we did with latitude and longitude, but with more values:

_Seinfeld_  vector: [Sitcom], 1989, 22-24, 9, 180  
_Wednesday_  vector: [Horror], 2022, 46-57, 1, 8

A machine learning model might identify the sitcom  _Cheers_  as being much more similar to  _Seinfeld_. It is of the same genre, debuted in 1982, features an episode length of 21-25 minutes, has 11 seasons, and has 275 episodes.

_Seinfeld_  vector: [Sitcom], 1989, 22-24, 9, 180  
_Cheers_  vector: [Sitcom], 1982, 21-25, 11, 275

In our examples above, a city was a point along the two dimensions of latitude and longitude; we then added a third dimension of population. We also analyzed the location of these TV shows along five dimensions.

Instead of two, three, or five dimensions, a TV show within a machine learning model is a point along perhaps a hundred or a thousand dimensions — however many the model wants to include.

## How do embeddings work?

Embedding is the process of creating vectors using  [deep learning](https://www.cloudflare.com/learning/ai/what-is-deep-learning/). An "embedding" is the output of this process — in other words, the vector that is created by a deep learning model for the purpose of similarity searches by that model.

![Embeddings - Document on left converted to vector with three dimensions on right by embeddings API](https://cf-assets.www.cloudflare.com/slt3lc6tev37/9Z8FkRhELv2fUEuEMEaFI/5ba681f6662c7ad4ab9a251a831a5781/document_becomes_embedding.png "What are embeddings? - example of embedding")

Embeddings that are close to each other — just as Seattle and Vancouver have latitude and longitude values close to each other and comparable populations — can be considered similar. Using embeddings, an algorithm can suggest a relevant TV show, find similar locations, or identify which words are likely to be used together or similar to each other, as in language models.

## How are embeddings used in large language models (LLMs)?

For LLMs, embedding is taken a step further. The context of every word becomes an embedding, in addition to the word itself. The meanings of entire sentences, paragraphs, and articles can be searched and analyzed. Although this takes quite a bit of computational power, the context for queries can be stored as embeddings, saving time and compute power for future queries.

ref: https://www.cloudflare.com/learning/ai/what-are-embeddings/
https://medium.com/the-ai-forum/rag-on-complex-pdf-using-llamaparse-langchain-and-groq-5b132bd1f9f3
