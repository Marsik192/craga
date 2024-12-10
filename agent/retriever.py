from langchain.agents.react.base import Tool

from agent.document_retrieval_chain import DocumentRetrievalChain
from data_pipeline.document_retriever import create_document_retriever
from data_pipeline.vectorstore import VectorStore
from agent.llm import LLM

RETRIEVER_TOOL_NAME = "retriever_tool"
RETRIEVER_TOOL_DESCRIPTION = """\
This tool allows you to retrieve relevant information from the provided documents.
If you want to use this tool, you MUST provide the following parameters in this format: \
{{"question" <query>, "long_question": <original user question>}}, where "question" is a detailed query to the vectorstore with important keywords, and \
"long_question" is the original question asked by the user. \
"""


class DocumentRetrieverTool:
    def __init__(self, vectorstore: VectorStore, num_documents=3, max_tokens_limit=4000):
        self.num_documents = num_documents
        self.max_tokens_limit = max_tokens_limit
        self.vectorstore = vectorstore
        self.retriever_tool = self.create_retriever_tool()

    def create_retriever_tool(self):
        """
        Create a tool to retrieve documents from vectorstore
        """

        llm = LLM()

        retriever = create_document_retriever(
            vectorstore=self.vectorstore,
            num_documents=self.num_documents
        )

        retrieval_chain = DocumentRetrievalChain.from_chain_type(
            llm=llm.model,
            retriever= retriever,
            max_tokens_limit=self.max_tokens_limit,
            verbose=True
        )

        retriever_tool = Tool(
            name=RETRIEVER_TOOL_NAME,
            func=retrieval_chain.invoke,
            description=RETRIEVER_TOOL_DESCRIPTION
        )

        return retriever_tool