
## What is Prompt Engineering?
The ability to provide a good starting point for the model and guide it to produce the right output plays a key role for applications that can integrate into daily work and make life easier.  **The output produced by language models varies significantly with the prompt served.**

**“Prompt Engineering”** is the practice of guiding the language model with a clear, detailed, well-defined, and optimized prompt in order to achieve a desired output.

There are two basic elements of a prompt. The language model needs a user-supplied instruction to generate a response. In other words, when a user provides an instruction, the language model produces a response.

## Prompt


![input data with output indicator](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.17.27-PM.png)

ref: Prompt Engineering  [Guide](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/lecture/Prompt-Engineering-Lecture-Elvis.pdf)

-   **Instructions:** This is the section where the task description is expressed. The task to be done must be clearly stated.
-   **Context:** A task can be understood differently depending on its context. For this reason, providing the command without its context can cause the language model to output something other than what is expected.
-   **Input data:** Indicates which and what kind of data the command will be executed on. Presenting it clearly to the language model in a structured format increases the quality of the response.
-   **Output indicator:** This is an indicator of the expected output. Here, what the expected output is can be defined structurally, so that output in a certain format can be produced.

## Types of Prompts

It is a well-known fact that **the better the prompt, the better the output!** So, what kinds of prompts are there? Let’s try to understand the different types of prompt! Before you know it, you’ll be a prompt engineer yourself!

Many advanced prompting techniques have been designed to improve performance on complex tasks, but first let’s get acquainted with simpler prompt types, starting with the most basic.

### Instruction Prompting

Simple instructions provide some guidance for producing useful outputs. For example, an instruction can express a clear and simple mathematical operation such as “Adding the numbers 1 to 99.”

![ChatGPT prompt 3](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.19.50-PM.png)

Or, you could try your hand at a slightly more complicated command. For example, maybe you want to analyze customer reviews for a restaurant separately according to taste, location, service, speed and price. You can easily do this with the command below:

![ChatGPT Prompt 4](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.20.59-PM.png)


![ChatGPT Prompt 6](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.22.17-PM.png)

### Role Prompting

Another approach is to assign a role to the artificial intelligence entity before the instructions. This technique generates somewhat more successful, or at least specific, outputs.

Now, let’s observe the difference when first assigning a role within the prompt. Let’s imagine a user who needs help to relieve tooth sensitivity to cold foods.

First, we try a simple command: **“I need help addressing my sensitivity to cold foods.”**

![role prompting](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.23.43-PM.png)


Now, let’s ask for advice again but this time we’ll assign the artificial intelligence a dentist role.

![ChatGPT Prompt 9](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.30.55-PM.png)

You can clearly see a difference in both the tone and content of the response, given the role assignment.
### “Standard” Prompting

Prompts are considered “standard” when they consist of only one question. For example, ‘Ankara is the capital of which country?’ would qualify as a standard prompt.

![standard prompting](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.32.29-PM.png)


#### **Few shot standard prompts**

Few shot standard prompts can be thought of as standard prompts in which a few samples are presented first. This approach is beneficial in that it facilitates learning in context. It is an approach that allows us to provide examples in the prompts to guide model performance and improvement.

![few shot standard prompts](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.34.22-PM.png)
### Chain of Thought (CoT) Prompting

Chain of Thought prompting is a way of simulating the reasoning process while answering a question, similar to the way the human mind might think it. If this reasoning process is explained with examples, the AI can generally achieve more accurate results.

![Comparison of models on the GSM8K benchmark](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.37.21-PM.png)

Comparison of models on the GSM8K benchmark

Now let’s try to see the difference through an example.

![Chain of Thought Prompting Elicits Reasoning in Large Language Models(2022)](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.38.21-PM.png)

Source:  [Chain of Thought Prompting Elicits Reasoning in Large Language Models(2022)](https://ai.googleblog.com/2022/05/language-models-perform-reasoning-via.html)

Above, an example of how the language model should think step-by-step is first presented to demonstrate how the AI should “think” through the problem or interpret it.

####  **“Zero Shot Chain of Thought (Zero-Shot CoT)”**

**“Zero Shot Chain of Thought (Zero-Shot CoT)”** slightly differentiates from this approach to prompt engineering. This time, it is seen that his reasoning ability can be increased again by adding a directive command like  **“Let’s think step by step”** without presenting an example to the language model.

![Zero Shot Chain of Thought](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.40.32-PM.png)

Source:  [Zero Shot Chain of Thought](https://github.com/dair-ai/Prompt-Engineering-Guide/blob/main/lecture/Prompt-Engineering-Lecture-Elvis.pdf)

In the experiments, it is seen that the **“Zero Shot Chain of Thought”** approach alone is not as effective as the Chain of Thought Prompting approach. On the other hand, it is of great importance what the redirect command is, and at this point, it has been observed that the “Let’s think step by step” command produces more successful results than many other commands.
## Recommendations and Tips for Prompt Engineering with OpenAI API

Let’s try to summarize some of OpenAI’s suggested tips and usage recommendations on how to give clear and effective instructions to GPT-3 and Codex when prompt engineering.

###### **Use latest models for the best results:**

If you are going to use it to generate text, the most current model is “text-davinci-003” and to generate code, it is “code-davinci-002” (November, 2022). You can [**check here**](https://platform.openai.com/docs/models/gpt-3) to follow the current models and for more detailed information about the models.

###### **Instructions must be at the beginning of the prompt, and the instruction and content must be separated by separators such as ### or “ “” :**

First of all, we must clearly state the instructions to the language model, and then use various separators to define the instruction and its content. Thus, it is presented to the language model in a more understandable way.

![Best practices with OpenAI](https://www.comet.com/site/wp-content/uploads/2023/04/Screenshot-2023-04-18-at-8.44.20-PM.png)

Source:  [Best practices for OpenAI](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

###### **Give instructions that are specific, descriptive and as detailed as possible:**

By typing clear commands on topics such as context, text length, format, style, you can get better outputs. For example, instead of an open-ended command like `Write a poem about OpenAI.` , you could write a more detailed command like `Write a short inspiring poem about OpenAI, focusing on the recent DALL-E product launch (DALL-E is a text to image ML model) in the style of a famous poet`

###### **Provide the output format expressed with examples:**

If you have a preferred output format in mind, we recommend providing a format example, as shown below:

**Less effective :**

Extract the entities mentioned in the text below. 
Extract the following 4 entity types: company names, people names, specific topics and themes.

Text: text

**Better :**

Extract the important entities mentioned in the text below. 
First extract all company names, then extract all people names, then extract specific topics which fit the content and finally extract general overarching themes

Desired format:
Company names: comma_separated_list_of_company_names
People names: -||-
Specific topics: -||-
General themes: -||-
Text: text

###### **Try zero-shot first, then continue with few-shot examples and fine-tune if you still don’t get the output you want:**

You can try zero-shot prompt engineering for your command without providing any examples to the language model. If you don’t get as successful output as you want, you can try few-shot methods by guiding the model with a few examples. If you still don’t produce as good an output as you intended, you can try fine-tuning.

There are examples of both zero-shot and few-shot prompts in the previous sections. You can check out  [**this best practices**](https://docs.google.com/document/d/1h-GTjNDDKPKU_Rsd0t1lXCAnHltaXTAzQ8K2HRhQf9U/edit) for fine-tune.

###### **Avoid imprecise explanations:**

When presenting a command to the language model, use clear and understandable language. Avoid unnecessary clarifications and details.

**Less effective :**

The description for this product should be fairly short, a few sentences only, 
and not too much more.

**Better :**

Use a 3 to 5 sentence paragraph to describe this product.

###### **Tell what to do rather than what not to do:**

Avoiding negative sentences and emphasizing intent will lead to better results.

**Less effective :**

The following is a conversation between an Agent and a Customer. DO NOT ASK USERNAME OR PASSWORD. DO NOT REPEAT.

Customer: I can't log in to my account.
Agent:

**Better :**

The following is a conversation between an Agent and a Customer. The agent will attempt to diagnose the problem and suggest a solution, whilst refraining from asking any questions related to PII. Instead of asking for PII, such as username or password, refer the user to the help article www.samplewebsite.com/help/faq

Customer: I can’t log in to my account.  
Agent:

###### **Code Generation Specific — Use “leading words” to nudge the model toward a particular pattern:**

It may be necessary to provide some hints to guide the language model when asking it to generate a piece of code. For example, a starting point can be provided, such as “import” that he needs to start writing code in Python, or “SELECT” when he needs to write an SQL query.

ref:https://www.comet.com/site/blog/prompt-engineering/
