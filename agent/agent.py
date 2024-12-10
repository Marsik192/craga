import asyncio

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

from agent.retriever import DocumentRetrieverTool
from agent.prompts import SYSTEM_MESSAGE, TOOLS, FORMAT_INSTRUCTIONS, SUFFIX 

from agent.llm import LLM


class AgentInterruptionError(Exception):
	pass


class Agent():
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key='input', output_key='output', return_messages=True
        )
        self.llm = LLM()
        
        self.vectorstore = None
        self.retriever = None
        self.tools = []

        self.update_vectorstore()

        self.prompt_template = self.create_prompt_template()

        self.agent = self.create_chat_agent()

        # Used for agent async calls
        self.agent_execution_task = None

    def create_prompt_template(self):
        """
        Create a prompt template based on the provider of the model.
        """
        template = SYSTEM_MESSAGE + TOOLS + FORMAT_INSTRUCTIONS + SUFFIX

        tool_messages = ""
        for tool in self.tools:
            msg = f"""
> {tool.name}
{tool.description}
            """
            tool_messages += msg

        prompt_template = PromptTemplate(
            partial_variables={
                "tools": "", # Required field, value is ignored
                "tool_messages": tool_messages,
            },
            input_variables=["chat_history", "input", "agent_scratchpad"],
            template=template
        )

        return prompt_template

    def create_chat_agent(self):
        """
        Create a conversational react agent
        """

        # Will be sent to LLM as an observation in case of parsing errors
        parsing_errors_observation = "Check your output: make sure it matches Action/Action Input syntax, or if it is a final output - make sure it has 'AI:' prefix"

        chat_agent = initialize_agent(
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            tools=self.tools,
            llm=self.llm.model,

            # Generate output in the case max_iterations is reached
            early_stopping_method="generate", 
            verbose=True,

            # Return chain of thought in agent's response
            return_intermediate_steps=True, 
            handle_parsing_errors=parsing_errors_observation,
            callbacks=[],
            max_iterations=2
        )

        # Assign a custom prompt
        chat_agent.agent.llm_chain.prompt = self.prompt_template

        return chat_agent

    def clean_memory(self):
        """
        Clear the memory of the agent.
        """
        self.agent.memory.clear()

    async def generate_response(self, input):
        """
        Generate a response from the agent.

        Args:
            input (str): The input to the agent.

        Returns:
            Returns the agent's response object.
        """

        # Create a coroutine and wrap it in a task
        coro = self.agent.ainvoke({
            "input": input,
            "chat_history": self.memory.buffer_as_messages,
        })
        self.agent_execution_task = asyncio.create_task(coro)

        try:
            response = await self.agent_execution_task
            self.agent_execution_task = None
        except asyncio.CancelledError:
            self.agent_execution_task = None
            raise AgentInterruptionError("Agent response interrupted")

        return response

    def interrupt_generation(self):
        """
        Interrupt agent's response.
        """

        if self.agent_execution_task and not self.agent_execution_task.done():
            self.agent_execution_task.cancel()
    
    def update_vectorstore(self, vectorstore=None):
        """
        
        """
        self.vectorstore = vectorstore

        if self.vectorstore is not None:

            self.retriever = DocumentRetrieverTool(self.vectorstore)

            self.tools = [
                self.retriever.retriever_tool
            ]
        else:
            self.retriever = None
            self.tools = []
        
        self.prompt_template = self.create_prompt_template()
        self.agent = self.create_chat_agent()