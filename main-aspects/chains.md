f## What are chains?
**A chain is an end-to-end wrapper around multiple individual components executed in a defined order.**

Chains are one of the core concepts of LangChain. Chains allow you to go beyond just a single API call to a language model and instead chain together multiple calls in a logical sequence.

They allow you to combine multiple components to create a coherent application.

**Some reasons you may want to use chains:**

-   To break down a complex task into smaller steps that can be handled sequentially by different models or utilities. This allows you to leverage the different strengths of different systems.
-   To add state and memory between calls. The output of one call can be fed as input to the next call to provide context and state.
-   To add additional processing, filtering or validation logic between calls.
-   For easier debugging and instrumentation of a sequence of calls.

## Foundational chain types in LangChain

The `LLMChain`, `RouterChain`, `SimpleSequentialChain`, and `TransformChain` are considered the core foundational building blocks that many other more complex chains build on top of. They provide basic patterns like chaining LLMs, conditional logic, sequential workflows, and data transformations.

• `LLMChain`: Chains together multiple calls to language models. Useful for breaking down complex prompts.

• `RouterChain`: Allows conditionally routing between different chains based on logic. Enables branching logic.

• `SimpleSequentialChain`: Chains together multiple chains in sequence. Useful for linear workflows.

• `TransformChain`: Applies a data transformation between chains. Helpful for data munging and preprocessing.

Other key chain types like `Agents` and `RetrievalChain` build on top of these foundations to enable more advanced use cases like goal-oriented conversations and knowledge-grounded generation.

However the foundational four provide the basic patterns for chain construction in LangChain.

### LLMChain

The most commonly used type of chain is an LLMChain.

The LLMChain consists of a PromptTemplate, a language model, and an optional output parser. For example, you can create a chain that takes user input, formats it with a PromptTemplate, and then passes the formatted response to an LLM. You can build more complex chains by combining multiple chains, or by combining chains with other components.

The main differences between using an LLMChain versus directly passing a prompt to an LLM are:

-   LLMChain allows chaining multiple prompts together, while directly passing a prompt only allows one. With LLMChain, you can break down a complex prompt into multiple more straightforward prompts and chain them together.
-   LLMChain maintains state and memory between prompts. The output of one prompt can be fed as input to the following prompt to provide context. Directly passing prompts lack this memory.
-   LLMChain makes adding preprocessing logic, validation, and instrumentation between prompts easier. This helps with debugging and quality control.
-   LLMChain provides some convenience methods like `apply` and `generate` that make it easy to run the chain over multiple inputs.

### Creating an LLMChain

To create an LLMChain, you need to specify:

-   The language model to use
-   The prompt template

### Code Example:

```
from langchain import PromptTemplate, OpenAI, LLMChain

# the language model
llm = OpenAI(temperature=0)

# the prompt template
prompt_template = "Act like a comedian and write a super funny two-sentence short story about {thing}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

llm_chain("A toddler hiding his dad's laptop")
```
```
{'thing': "A toddler hiding his dad's laptop",
 'text': '\n\nThe toddler thought he was being sneaky, but little did he know his dad was watching the whole time from the other room, laughing.'}
```
Use `apply` when you have a list of inputs and want to get the LLM to generate text for each one, it will run the LLMChain for every input dictionary in the list and return a list of outputs.

```
input_list = [
    {"thing": "a Punjabi rapper who eats too many samosas"},
    {"thing": "a blind eye doctor"},
    {"thing": "a data scientist who can't do math"}
]

llm_chain.apply(input_list)
```

```
[{'text': "\n\nThe Punjabi rapper was so famous that he was known as the 'Samosa King', but his fame was short-lived when he ate so many samosas that he had to be hospitalized for a stomachache!"},
 {'text': "\n\nA blind eye doctor was so successful that he was able to cure his own vision - but he still couldn't find his glasses."},
 {'text': '\n\nA data scientist was so bad at math that he had to hire a calculator to do his calculations for him. Unfortunately, the calculator was even worse at math than he was!'}]
 ```

`generate` is similar to apply, except it returns an `LLMResult` instead of a string. Use this when you want the entire `LLMResult` object returned, not just the generated text. This gives you access to metadata like the number of tokens used.

```
llm_chain.generate(input_list)
```

```
LLMResult(generations=

[[Generation(text="\n\nThe Punjabi rapper was so famous that he was known as the 'Samosa King', 
but his fame was short-lived when he ate so many samosas that he had to be hospitalized for a stomachache!", 
generation_info={'finish_reason': 'stop', 'logprobs': None})], 

[Generation(text="\n\nA blind eye doctor was so successful that he was able to cure his own vision - but he still couldn't find his glasses.", generation_info={'finish_reason': 'stop', 'logprobs': None})], 

[Generation(text='\n\nA data scientist was so bad at math that he had to hire a calculator to do his calculations for him. Unfortunately, the calculator was even worse at math than he was!', generation_info={'finish_reason': 'stop', 'logprobs': None})]],

llm_output={'token_usage': {'prompt_tokens': 75, 'total_tokens': 187, 'completion_tokens': 112}, 'model_name': 'text-davinci-003'}, run=[RunInfo(run_id=UUID('b638d2c6-77d9-4346-8494-866892e36bc5')), RunInfo(run_id=UUID('427f9e51-4848-49d3-83c1-e96131f2b34f')), RunInfo(run_id=UUID('4201eea9-1616-42e7-8cb2-a5b26128decd'))])
```
Use `predict` when you want to pass inputs as keyword arguments instead of a dictionary. This can be convenient if you don’t want to construct an input dictionary.

```
llm_chain.predict(thing="colorful socks")
```
```
The socks were so colorful that when the washing machine finished its cycle, the socks had formed a rainbow in the laundry basket!
```
Use `LLMChain.run` when you want to pass the input as a dictionary and get the raw text output from the LLM.

`LLMChain.run` is convenient when your LLMChain has a single input key and a single output key.

```
llm_chain.run("the red hot chili peppers")
```
```
['1. Wear a Hawaiian shirt\n2. Sing along to the wrong lyrics\n3. Bring a beach ball to the concert\n4. Try to start a mosh pit\n5. Bring a kazoo and try to join in on the music']
```

### Parsing output

To parse the output, you simply pass an output parser directly to `LLMChain`.

```
from langchain.output_parsers import CommaSeparatedListOutputParser

llm = OpenAI(temperature=0)

# the prompt template
prompt_template = "Act like a Captain Obvious and list 5 funny things to not do at {place}?"

output_parser=CommaSeparatedListOutputParser()

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template),
    output_parser= output_parser
)

llm_chain.predict(place='Disneyland')
```

```
['1. Wear a costume of a Disney villain.\n2. Bring your own food and drinks into the park.\n3. Try to ride the roller coasters without a ticket.\n4. Try to sneak into the VIP area.\n5. Try to take a selfie with a Disney character without asking permission.']
```


## Router Chains

Router chains allow routing inputs to different destination chains based on the input text. This allows the building of chatbots and assistants that can handle diverse requests.

-   Router chains examine the input text and route it to the appropriate destination chain
-   Destination chains handle the actual execution based on the input
-   **Router chains are powerful for building multi-purpose chatbots/assistants**

The following example will show routing chains used in a `MultiPromptChain` to create a question-answering chain that **selects the prompt which is most relevant for a given question and then answers the question using that prompt**.

```
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "physics",
        "description": "Good for answering questions about physics",
        "prompt_template": physics_template,
    },
    {
        "name": "math",
        "description": "Good for answering math questions",
        "prompt_template": math_template,
    },
]

destination_chains = {}

for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = chain

default_chain = ConversationChain(llm=llm, output_key="text")

default_chain.run("What is math?")
```
```
 Math is the study of numbers, shapes, and patterns. It is used to solve problems and understand the world around us. It is a fundamental part of our lives and is used in many different fields, from engineering to finance.
 ```
## Sequential Chains

Sometimes, you might want to make a series of calls to a language model, take the output from one call and use it as the input to another. Sequential chains allow you to connect multiple chains and compose them into pipelines executing a specific scenario.

There are two types of sequential chains:

1) `SimpleSequentialChain`: The simplest form of sequential chains, where each step has a singular input/output, and the output of one step is the input to the next.

2) `SequentialChain`: A more general form of sequential chains allows multiple inputs/outputs.

### SimpleSequentialChain

The simplest form of a sequential chain is where each step has a single input and output.

The output of one step is passed as input to the next step in the chain. You would use `SimpleSequentialChain` it when you have a linear pipeline where each step has a single input and output. `SimpleSequentialChain` implicitly passes the output of one step as input to the next.

This is great for composing a precise sequence of LLMChains where each builds directly on the previous output.

### When to use:

-   You have a clear pipeline of steps, each with a single input and output
-   Each step builds directly off the previous step’s output
-   Useful for simple linear pipelines with one input and output per step.
-   Create each step as an `LLMChain`.
-   Pass list of `LLMChains` to `SimpleSequentialChain`.
-   Call `run()` passing the initial input.

### How to use:

1) Define each step as an `LLMChain` with a single input and output

2) Create a `SimpleSequentialChain` passing a list of the LLMChain steps

3) Call `run()` on the SimpleSequentialChain with the initial input

```
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# This is an LLMChain to write a rap.
llm = OpenAI(temperature=.7)

template = """

You are a Punjabi Jatt rapper, like AP Dhillon or Sidhu Moosewala.

Given a topic, it is your job to spit bars on of pure heat.

Topic: {topic}
"""
prompt_template = PromptTemplate(input_variables=["topic"], template=template)

rap_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a diss track

llm = OpenAI(temperature=.7)

template = """

You are an extremely competitive Punjabi Rapper.

Given the rap from another rapper, it's your job to write a diss track which
tears apart the rap and shames the original rapper.

Rap:
{rap}
"""

prompt_template = PromptTemplate(input_variables=["rap"], template=template)

diss_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SimpleSequentialChain

overall_chain = SimpleSequentialChain(chains=[rap_chain, diss_chain], verbose=True)

review = overall_chain.run("Drinking Crown Royal and mobbin in my red Challenger")
```

### SequentialChain

A more general form of sequential chain allows multiple inputs and outputs per step.

You would use `SequentialChain` when you have a more complex pipeline where steps might have multiple inputs and outputs.

`SequentialChain` allows you to explicitly specify all the input and output variables at each step and map outputs from one step to inputs of the next. This provides more flexibility when steps might have multiple dependencies or produce multiple results to pass along.

### When to use:

-   You have a sequence of steps but with more complex input/output requirements
-   You need to track multiple variables across steps in the chain

### How to use

-   Define each step as an LLMChain, specifying multiple input/output variables
-   Create a SequentialChain specifying all input/output variables
-   Map outputs from one step to inputs of the next
-   Call run() passing a dict of all input variables
-   The key difference is `SimpleSequentialChain` handles implicit variable passing whereas SequentialChain allows explicit variable specification and mapping.

### When you would use SequentialChain vs SimpleSequentialChain

Use `SimpleSequentialChain` for linear sequences with a single input/output. Use `SequentialChain` for more complex sequences with multiple inputs/outputs.

### The key difference

`SimpleSequentialChain` is for linear pipelines with a single input/output per step. Implicitly passes variables.

`SequentialChain` handles more complex pipelines with multiple inputs/outputs per step. Allows explicitly mapping variables.

This uses a standard ChatOpenAI model and prompt template. You chain them together with the `|` operator and then call it with `chain.invoke`. We can also get async, batch, and streaming support out of the box.

```
llm = OpenAI(temperature=.7)

template = """

You are a Punjabi Jatt rapper, like AP Dhillon or Sidhu Moosewala.

Given two topics, it is your job to create a rhyme of two verses and one chorus
for each topic.

Topic: {topic1} and {topic2}

Rap:

"""

prompt_template = PromptTemplate(input_variables=["topic1", "topic2"], template=template)

rap_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="rap")


template = """

You are a rap critic from the Rolling Stone magazine and Metacritic.

Given a, it is your job to write a review for that rap.

Your review style should be scathing, critical, and no holds barred.

Rap:

{rap}

Review from the Rolling Stone magazine and Metacritic critic of the above rap:

"""

prompt_template = PromptTemplate(input_variables=["rap"], template=template)

review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is the overall chain where we run these two chains in sequence.
from langchain.chains import SequentialChain

overall_chain = SequentialChain(
    chains=[rap_chain, review_chain],
    input_variables=["topic1", "topic2"],
    # Here we return multiple variables
    output_variables=["rap", "review"],
    verbose=True)

overall_chain({"topic1":"Tractors and sugar canes", "topic2": "Dasuya, Punjab"})
```

```
> Entering new SequentialChain chain...

> Finished chain.
{'topic1': 'Tractors and sugar canes',
 'topic2': 'Dasuya, Punjab',
 'rap': "Verse 1\nI come from a place with lots of fame\nDasuya, Punjab, where the tractors reign\nI'm a Jatt rapper with a game to play\nSo I'm gonna take it up and make it my way\n\nChorus\nTractors and sugar canes, that's what I'm talking about\nTractors and sugar canes, it's all about\nDasuya, Punjab, a place so grand\nTractors and sugar canes, that's our jam\n\nVerse 2\nFrom Punjab's beauty I derive my pride\nMy heart belongs to the place, where the sugar canes reside\nWhere the soil is my home, I'm never apart\nFrom the tractors and sugar canes of Dasuya, Punjab\n\nChorus\nTractors and sugar canes, that's what I'm talking about\nTractors and sugar canes, it's all about\nDasuya, Punjab, a place so grand\nTractors and sugar canes, that's our jam",
 'review': "\nThis rap artist hails from the small town of Dasuya, Punjab, and takes pride in his hometown's culture and agricultural way of life. While the lyrical content of this rap is filled with references to tractors and sugar canes, unfortunately the artist's delivery falls flat and fails to capture the unique essence of his home. The basic rhyme scheme, repetitive chorus, and lack of originality make this a forgettable track. The artist's enthusiasm for his hometown is admirable, but unfortunately it is not enough to make this rap stand out from the crowd."}
 ```
 
 ## Transformation

Transformation Chains allows you to define custom data transformation logic as a step in your LangChain pipeline. This is useful when you must preprocess or transform data before passing it to the next step.
```
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

!wget https://www.gutenberg.org/files/2680/2680-0.txt

with open("/content/2680-0.txt") as f:
    meditations = f.read()

def transform_func(inputs: dict) -> dict:
    """
    Extracts specific sections from a given text based on newline separators.

    The function assumes the input text is divided into sections or paragraphs separated
    by one newline characters (`\n`). It extracts the sections from index 922 to 950
    (inclusive) and returns them in a dictionary.

    Parameters:
    - inputs (dict): A dictionary containing the key "text" with the input text as its value.

    Returns:
    - dict: A dictionary containing the key "output_text" with the extracted sections as its value.
    """
    text = inputs["text"]
    shortened_text = "\n".join(text.split("\n")[921:950])
    return {"output_text": shortened_text}

transform_chain = TransformChain(
    input_variables=["text"], output_variables=["output_text"], transform=transform_func, verbose=True
)

transform_chain.run(meditations)
```
```
II. Let it be thy earnest and incessant care as a Roman and a man to
perform whatsoever it is that thou art about, with true and unfeigned
gravity, natural affection, freedom and justice: and as for all other
cares, and imaginations, how thou mayest ease thy mind of them. Which
thou shalt do; if thou shalt go about every action as thy last action,
free from all vanity, all passionate and wilful aberration from reason,
and from all hypocrisy, and self-love, and dislike of those things,
which by the fates or appointment of God have happened unto thee. Thou
seest that those things, which for a man to hold on in a prosperous
course, and to live a divine life, are requisite and necessary, are not
many, for the gods will require no more of any man, that shall but keep
and observe these things.

III. Do, soul, do; abuse and contemn thyself; yet a while and the time
for thee to respect thyself, will be at an end. Every man's happiness
depends from himself, but behold thy life is almost at an end, whiles
affording thyself no respect, thou dost make thy happiness to consist in
the souls, and conceits of other men.

IV. Why should any of these things that happen externally, so much
distract thee? Give thyself leisure to learn some good thing, and cease
roving and wandering to and fro. Thou must also take heed of another
kind of wandering, for they are idle in their actions, who toil and
labour in this life, and have no certain scope to which to direct all
their motions, and desires. V. For not observing the state of another
man's soul, scarce was ever any man known to be unhappy. Tell whosoever
they be that intend not, and guide not by reason and discretion the
motions of their own souls, they must of necessity be unhappy.
```

```
template = """

Rephrase this text:

{output_text}

In the style of a 90s gangster rapper speaking to his homies.

Rephrased:"""

prompt = PromptTemplate(input_variables=["output_text"], template=template)

llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain], verbose=True)

sequential_chain.run(meditations)
```

```
> Entering new SimpleSequentialChain chain...


> Entering new TransformChain chain...

> Finished chain.

II. Let it be thy earnest and incessant care as a Roman and a man to
perform whatsoever it is that thou art about, with true and unfeigned
gravity, natural affection, freedom and justice: and as for all other
cares, and imaginations, how thou mayest ease thy mind of them. Which
thou shalt do; if thou shalt go about every action as thy last action,
free from all vanity, all passionate and wilful aberration from reason,
and from all hypocrisy, and self-love, and dislike of those things,
which by the fates or appointment of God have happened unto thee. Thou
seest that those things, which for a man to hold on in a prosperous
course, and to live a divine life, are requisite and necessary, are not
many, for the gods will require no more of any man, that shall but keep
and observe these things.

III. Do, soul, do; abuse and contemn thyself; yet a while and the time
for thee to respect thyself, will be at an end. Every man's happiness
depends from himself, but behold thy life is almost at an end, whiles
affording thyself no respect, thou dost make thy happiness to consist in
the souls, and conceits of other men.

IV. Why should any of these things that happen externally, so much
distract thee? Give thyself leisure to learn some good thing, and cease
roving and wandering to and fro. Thou must also take heed of another
kind of wandering, for they are idle in their actions, who toil and
labour in this life, and have no certain scope to which to direct all
their motions, and desires. V. For not observing the state of another
man's soul, scarce was ever any man known to be unhappy. Tell whosoever
they be that intend not, and guide not by reason and discretion the
motions of their own souls, they must of necessity be unhappy.


Yo, listen up my homies, it's time to get serious. We gotta take care of our business and act with true gravity, natural affection, freedom, and justice. So forget all those other cares and worries, and just do every action like it's your last, stayin' away from vanity and all that phony stuff. We don't need much for true happiness. All the gods ask is that we keep it real and show some respect for ourselves. Don't let nothin' from the outside distract you. Take time to learn something good and make sure you got a goal to get to. Don't worry 'bout anybody else, 'cause if you don't look after your own soul, you gonna end up real unhappy.

> Finished chain.
\n\nYo, listen up my homies, it's time to get serious. We gotta take care of our business and act with true gravity, natural affection, freedom, and justice. So forget all those other cares and worries, and just do every action like it's your last, stayin' away from vanity and all that phony stuff. We don't need much for true happiness. All the gods ask is that we keep it real and show some respect for ourselves. Don't let nothin' from the outside distract you. Take time to learn something good and make sure you got a goal to get to. Don't worry 'bout anybody else, 'cause if you don't look after your own soul, you gonna end up real unhappy.
```

ref: https://www.comet.com/site/blog/chaining-the-future-an-in-depth-dive-into-langchain/

