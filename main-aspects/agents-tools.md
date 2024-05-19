As a brief:

*Agents* are like characters or personas with specific capabilities. They use chains and tools to perform their functions.
*Chains* are sequences of processing steps for prompts. They are used within agents to define how the agent processes information.
*Tools* are specialized functionalities that can be used by agents or within chains for specific tasks.


## Tools

Tools are functions that agents can use to interact with the world. They are functions that are supposed to perform specific duties. These tools can be generic utilities (e.g. google search, database lookups, mathematical opeartions etc.), other chains, or even other agents.

Tools allow for the LLM to interact with the outside world and since they are customizable they can pretty much coded to do anything you like and not just some limited pre-defined operations.

## **Agents**

Some applications will require not just a predetermined chain of calls to LLMs/other tools, but potentially an unknown chain that depends on the user’s input. In these types of chains, there is a “agent” which has access to a suite of tools. Depending on the user input, the agent can then decide which, if any, of these tools to call.

The core idea of agents is to use an LLM to choose a sequence of actions to take. In chains, a sequence of actions is hardcoded (in code). In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

Simply put,  **Agent = Tools + Memory**

![](https://miro.medium.com/v2/resize:fit:700/1*S7e0jWcLVxR2583BugkQuQ.png)

Looking at the diagram below, when receiving a request, Agents make use of a LLM to decide on which Action to take.

After an Action is completed, the Agent enters the Observation step.

From Observation step Agent shares a Thought; if a final answer is not reached, the Agent cycles back to another Action in order to move closer to a Final Answer.

There is a whole array of Action options available to the LangChain Agent.

Actions are taken by the agent via various tools. The more tools are available to an Agent, the more actions can be taken by the Agent.

There are many types of agents such as — Conversations, ReAct etc. Custom agents can be made as well.
## Chains

Using an LLM in isolation is fine for simple applications, but more complex applications require chaining LLMs — either with each other or with other components.

LangChain provides the Chain interface for such “chained” applications. We define a Chain very generically as a sequence of calls to components, which can include other chains.

In the sample project explained in this article, the Sequential Chain is used which will give very clear insight into how these chains work.

Langchain has 4 types of foundational chains -

1.  **LLM**  — A simple chain with a prompt template that can process multiple inputs.
2.  **Router**  — A gateway that uses the large language model (LLM) to select the most suitable processing chain.
3.  **Sequential**  — A family of chains which processes input in a sequential manner. This means that the output of the first node in the chain, becomes the input of the second node and the output of the second, the input of the third and so on.
4.  **Transformation**  — A type of chain that allows Python function calls for customizable text manipulation.

## **Memory**

You can provide attach memory to your so that it remembers the context of the conversation and responds accordingly.

1.  **Buffer Memory:** The Buffer memory in Langchain is a simple memory buffer that stores the history of the conversation. It has a buffer property that returns the list of messages in the chat memory. The load_memory_variables function returns the history buffer. This type of memory is useful for storing and retrieving the immediate history of a conversation.
2.  **Buffer Window Memory:**  Buffer Window Memory is a variant of Buffer Memory. It also stores the conversation history but with a twist. It has a property k which determines the number of previous interactions to be stored. The buffer property returns the last k*2 messages from the chat memory. This type of memory is useful when you want to limit the history to a certain number of previous interactions.
3.  **Entity Memory:**  The Entity Memory in Langchain is a more complex type of memory. It not only stores the conversation history but also extracts and summarizes entities from the conversation. It uses the Langchain Language Model (LLM) to predict and extract entities from the conversation. The extracted entities are then stored in an entity store which can be either in-memory or Redis-backed. This type of memory is useful when you want to extract and store specific information from the conversation.

Each of these memory types has its own use cases and trade-offs. Buffer Memory and Buffer Window Memory are simpler and faster but they only store the conversation history. Entity Memory, on the other hand, is more complex and slower but it provides more functionality by extracting and summarizing entities from the conversation.

As for the data structures and algorithms used, it seems that Langchain primarily uses lists and dictionaries to store the memory. The algorithms are mostly related to text processing and entity extraction, which involve the use of the Langchain Language Model.

1.  **Conversation Knowledge Graph Memory:** The Conversation Knowledge Graph Memory is a sophisticated memory type that integrates with an external knowledge graph to store and retrieve information about knowledge triples in the conversation. It uses the Langchain Language Model (LLM) to predict and extract entities and knowledge triples from the conversation. The extracted entities and knowledge triples are then stored in a NetworkxEntityGraph, which is a type of graph data structure provided by the NetworkX library. This memory type is useful when you want to extract, store, and retrieve structured information from the conversation in the form of a knowledge graph.
2.  **ConversationSummaryMemory:**  The ConversationSummaryMemory is a type of memory that summarizes the conversation history. It uses the LangChain Language Model (LLM) to generate a summary of the conversation. The summary is stored in a buffer and is updated every time a new message is added to the conversation. This memory type is useful when you want to maintain a concise summary of the conversation that can be used for reference or to provide context for future interactions.
3.  **ConversationSummaryBufferMemory:** ConversastionSummaryBufferMemory is similar to the ConversationSummaryMemory but with an added feature of pruning. If the conversation becomes too long (exceeds a specified token limit), the memory prunes the conversation by summarizing the pruned part and adding it to a moving summary buffer. This ensures that the memory does not exceed its capacity while still retaining the essential information from the conversation.
4.  **ConversationTokenBufferMemory:**  ConversationTokenBufferMemory is a type of memory that stores the conversation history in a buffer. It also has a pruning feature similar to the ConversationSummaryBufferMemory. If the conversation exceeds a specified token limit, the memory prunes the earliest messages until it is within the limit. This memory type is useful when you want to maintain a fixed-size memory of the most recent conversation history.
5.  **VectorStore-Backed Memory:**  The VectorStore-Backed Memory is a memory type that is backed by a VectorStoreRetriever. The VectorStoreRetriever is used to retrieve relevant documents based on a query. The retrieved documents are then stored in the memory. This memory type is useful when you want to store and retrieve information in the form of vectors, which is particularly useful for tasks such as semantic search or similarity computation.

## Callback Handlers

LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. This is useful for logging, monitoring, streaming, and other tasks. The BaseCallbackHandler class is used to define the actions to be performed inside the hook functions.

Some available hooks are — on_llm_start, on_agent_end, on_chain_start. The names of these hooks are self explanatory. Code can be written inside these functions which has to be performed when those functions are called.

The object of the BaseCallbackHandler class can provided to the appropriate agent, chain, tool etc.

# **Walkthrough — Project Utilizing Langchain**

The following image displays the architecture I’ve used in a project that helps in answering questions on data available in a large SQL database, by creating SQL queries to fetch relevant data, then analyzing the fetched data and then returning a response in the form of answer.

![](https://miro.medium.com/v2/resize:fit:700/1*vDt2qXB5W7AWZAXFkjWwgA.png)

In the image above it can be seen that the agent has two chains available to it as tools which are -

1.  Analysis Chain (For doing analysis on data in memory)
2.  Sequential Chain (For writing SQL queries)

**NOTE: While there are predefined and configured agents, tools and chains are available, custom versions of all of these can be made.**

**NOTE: Chains can be provided as tools to the agent. Similarly, Tools can be made available as a chain segment in chains as well. The user has a lot of freedom to customize these agents, tools, chains and can plug, sequence them according to their needs.**

tools=[  
    Tool.from_function(func=sequentialchain._run,  
                    name="tool1",  
                    description="Useful when user wants information about revenue, margin, employee and projects. Input is a descriptive plain text formed using user question and chat history and output is the result."  
    ),  
      
    Tool.from_function(func=analysis._run,  
                    name="tool2",  
                    description="Useful when you want to do some calculations and statistical analysis using the memory. Input is a list of numbers with description of what is to be done to it or a mathematical equation of number and output is result."  
    )  
    ]

The code snippet above shows a tools array in which two chains, namely — sequentialchain and analysis chain are provided as tools.
```
memory = ConversationBufferWindowMemory(memory_key="chat_history",return_messages=True,k=7)  
llm = AzureChatOpenAI(  
    temperature=0,  
    deployment_name="********************",  
    model_name="gpt-35-turbo-16k",  
    openai_api_base="***************************",  
    openai_api_version="2023-07-01-preview",  
    openai_api_key="**************",  
    openai_api_type="azure"  
)  
agent_chain=initialize_agent(  
    tools,  
    llm,  
    agent=AgentType.OPENAI_FUNCTIONS,  
    verbose=True,  
    agent_kwargs=agent_kwargs,  
    memory=memory,  
    callbacks=[MyCustomHandler()]  
)
```
The initialize_agent function creates an agent object with the specifications you have entered in the function as arguments.

This agent is what manages the whole interaction with the LLM. The agent is run like this → answer=agent_chain.run(“the query put in by the user”)

The tools and memory are provided to the agent. I have used the ConversationBufferWindowMemory() which allows me to specify the value k as 7. This means that the last 7 conversations (input and output) are available to the LLM when you ask a new question.

class sequentialchain(BaseTool):  
     def _run( self, run_manager: Optional[CallbackManagerForToolRun] = None ) -> str:  
        tables = similarity_search(self)  
        print(tables)  
        sql_chain  = SQLAgent(tables)  
        querycheckchain=querycheckfunc(tables)  
        executorchainobj=QueryExecutorChain(user_query=self)  
        overall_chain = SimpleSequentialChain(chains=[sql_chain, querycheckchain, executorchainobj], verbose=True)  
        review = overall_chain.run(self)  
        return review

The similarity_search() function gets the appropriate table descriptions from the vector db and provides it as input variables for the chains so they can write proper SQL queries.

The SimpleSequentialChain() has 3 chains passed to it — sql_chain, querycheckchain, executorchainobj which are run in succession. The output of the first chain is passed to the second chain as an input variable and the output of the third chain is passed to the third chain as an input variable.

The  **sql_chain** — based on a prompt on how to create SQL queries and table descriptions makes SQL queries.

The **querycheckchain** — Receives the SQL query from sql_chain, then corrects all the errors, syntax, adds missing elements if any and makes it compliant to the standards described in prompt.

The  **executorchainobj** — This chain segment is actually a tool passed as a chain. It Receives the SQL query that is ready to be run on the database.

The output or fetched data after running the SQL query is then received by the agent which had called the sequentialchain. The agent would interpret the fetched in accordance to the user’s input question, format it and provide the final answer/response to the user. If the agent wants to do some analysis on the fetched data it can then send this data to the analysis chain the output of which can then be formatted into a final answer/response.

If the question asked by a user is a follow up question, the agent can look at the memory and if it can find the necessary data in it, then it can formulate the answer based on the memory alone as well, or if it thinks some analysis is to be done then it can also directly send that data to the analysis chain as well.

Agent decides when to use the memory, which tool to use or if to use any tool at all.

**NOTE: I have used a custom chain (analysis chain) provided as a tool to the agent. There are predefined tools for all sorts of purposes like math, SQL connections, google drive connections, AWS Lambda connections etc.**

The analysis chain is a normal LLM call chain and has prompt instructions to do various types of statistical analysis (mean, median, standard deviation, variance etc.), calculate growth, percentages and other mathematical operations.

Callback Handlers can also be added to perform various tasks at certain defined stages of the application run cycle.
```
class MyCustomHandler(BaseCallbackHandler):  
    def on_llm_new_token(self, token: str, **kwargs) -> None:  
        print(f"My custom handler, token: {token}")  
        for key, value in kwargs.items():  
            print("%s == %s" % (key, value))  
  
    def on_llm_end( self,  
        outputs,  
        *,  
        run_id,  
        parent_run_id,  
        **kwargs, ):  
        """Run when llm call ends running."""  
        print(run_id)  
  
  
    def on_chain_end( self,  
        outputs,  
        *,  
        run_id,  
        parent_run_id,  
        **kwargs, ):  
        """Run when chain ends running."""  
        print(run_id)
```
The CallbackHandler — MyCustomHandler() has been configured with certains set of code that would run on — on_chain_end, on_llm_end. The names of these hooks are self explanatory. When the object of this class is provided to the appropriate agent, tool, chain etc., these code inside these hooks would run as their names suggest.

All sorts of hooks such as on_chain_start, on_chain_end, on_tool_start, on_tool_end are available which can be specified to do certain tasks under the BaseCallbackHandler Class.
```
prompt_template = PromptTemplate(input_variables=["query"], template=template)  
query_check_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review", callbacks=[MyCustomHandler()])
```
The hook in this case — MyCustomHandler(), can be provided to the appropriate agent, tool or chain in the callbacks argument.

When all of this is set up when the agent is run — (agent_chain,run(“user’s input question”)), the application can self write the sql queries, run them to fetch data from the database, analyse the data, and give proper information as output to the user. The user never has to even open the database, write sql queries, fetch the data, dig through it for analysis etc. Everything happens automatically from start to finish.

ref: https://medium.com/@saumitra1joshi/langchain-agents-tools-chains-memory-for-utilizing-the-full-potential-of-llms-211e5dfee3fa

https://community.deeplearning.ai/t/agents-vs-chains-vs-tools/516148/2
