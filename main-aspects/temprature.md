
# Top P and Temperature

Large Language Models(LLMs) are essential tools in natural language processing (NLP) and have been used in a variety of applications, such as text completion, translation, and question answering.

The output of large language models can be affected by various hyperparameters including temperature, top p, token length, max tokens and stop tokens.

## Temperature

Temperature is a hyperparameter that controls the randomness of language model output.

A high temperature produces more unpredictable and creative results, while a low temperature produces more deterministic and conservative output. In other words, a higher temperature setting causes the model to be more “confident” in its output. A lower temperature setting yields more conservative and predictable output.

For example, if you adjust the temperature to 0.5, the model will generate text that is more predictable and less creative than if you set the temperature to 1.0.

temperature: Controls the randomness of responses. A lower temperature leads to more predictable outputs, while a higher temperature results in more varied and sometimes more creative outputs

## Top p

Top p, also known as nucleus sampling, is another hyperparameter that controls the randomness of language model output.

It sets a threshold probability and selects the top tokens whose cumulative probability exceeds the threshold. The model then randomly samples from this set of tokens to generate output. This method can produce more diverse and interesting output than traditional methods that randomly sample the entire vocabulary.

For example, if you set top p to 0.9, the model will only consider the most likely words that make up 90% of the probability mass.

top_p: can be considere as a method of text generation that selects the next token from the probability distribution of the top p most likely tokens. This balances exploration and exploitation during generation

## Token length

This is the number of words or characters in a sequence or text that is fed to the LLM.

It varies depending on the language and the tokenization method used for the particular LLM.

The length of the input text affects the output of the LLM.

A very short input may not have enough context to generate a meaningful completion.

Conversely, a rather long input may make the model inefficiently process or it may cause the model to generate an irrelevant output.

## Max tokens

This is the maximum number of tokens that the LLM generates.

Within this, is the token limit; the maximum number of tokens that can be used in the prompt and the completion of the model. Determined by the architecture of the model LLM, it refers to the maximum tokens that can be processed at once.

The computational cost and the memory requirements are directly proportional to the max tokens. Set a longer max token, and you will have greater context and coherent output text. Set a shorter max token, and you will use less memory and have a faster response but your output is prone to errors and inconsistencies.

During the training and fine-tuning of the LLM, the max token is set.

Contrary to fine-tuning token length during the generation of output, the coherence and length of the output is carefully set at inception, based on the specific task & requirements, without affecting other parameters that will likely need adjusting.

max_tokens: The maximum number of tokens that the model can process in a single response. This limit ensures computational efficiency and resource management

## Stop tokens

In simple terms, it is the length of the output or response of an LLM.

So it signifies the end of a sequence in terms of either a paragraph or a sentence.

Similar to max tokens, the inference budget is reduced when the stop tokens are set low.

For example, when the stop tokens are set at 2, the generated text or output will be limited to a paragraph. If the stop tokens is set at 1, the generated text will be limited to a sentence.

ref: https://medium.com/@dixnjakindah/top-p-temperature-and-other-parameters-1a53d2f8d7d7
