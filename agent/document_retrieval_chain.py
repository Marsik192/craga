import ast
import re
from typing import Tuple
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate


RETRIEVAL_CHAIN_PROMPT_TEMPLATE = """Given the following extracted parts of a long document and a question, create a well-detailed answer with as much numerical data and arguments as possible with references ("SOURCES"). Consider "LONG QUESTION" to understand the original question.

If you don't know the answer, just say that you don't know. Don't try to make up an answer.

ALWAYS return a "SOURCES" part in your answer, use the following format when returning sources: "SOURCES: <source_1>, <source_2>, <source_n>". For example: "SOURCES: Documentation.pdf, data_orders.xlsx".

QUESTION: What are the key differences between TCP and UDP protocols?
=========
Content: TCP (Transmission Control Protocol) is a connection-oriented protocol that provides reliable data transfer with error checking and flow control mechanisms. It ensures that data packets are delivered in order and retransmits lost packets.
Source: networking_fundamentals.pdf
Content: UDP (User Datagram Protocol) is a connectionless protocol that does not guarantee delivery, order, or error checking, making it faster but less reliable than TCP.
Source: networking_fundamentals.pdf
Content: TCP is used for applications where reliability is crucial, such as web browsing (HTTP/HTTPS), email (SMTP), and file transfer (FTP).
Source: protocol_comparison.docx
Content: UDP is used in applications where speed is more critical than reliability, such as video streaming, online gaming, and VoIP.
Source: protocol_comparison.docx
=========
FINAL ANSWER: The key differences between TCP and UDP protocols are:
- TCP is connection-oriented, establishing a connection before data transfer, while UDP is connectionless.
- TCP provides reliable data transfer with error checking, acknowledgment, and retransmission of lost packets. UDP does not guarantee delivery or order.
- TCP is slower due to overhead from connection management and error handling. UDP is faster with minimal overhead.
- TCP is used for applications requiring reliability (e.g., web browsing, email, file transfers), whereas UDP is used for applications where speed is essential (e.g., video streaming, online gaming).

SOURCES: networking_fundamentals.pdf, protocol_comparison.docx

QUESTION: Які основні функції останньої версії програмного продукту ABC?
=========
Content: The latest version of ABC software introduces a new artificial intelligence module that enhances data analytics capabilities.
Source: abc_release_notes.pdf
Content: It also includes a revamped user interface for better user experience and supports integration with various third-party services through API enhancements.
Source: abc_release_notes.pdf
Content: Ключові оновлення в цій версії:
- AI Модуль: Реалізує алгоритми машинного навчання для прогнозної аналітики.
- Інтерфейс Користувача: Оновлений дизайн з налаштовуваними панелями управління.
- Продуктивність: Оптимізована кодова база, що призводить до збільшення швидкості обробки на 40%.
- Безпека: Додана багатофакторна аутентифікація та шифрування даних у стані спокою.
Source: abc_user_manual_ua.docx
=========
FINAL ANSWER: Остання версія програмного продукту ABC включає модуль штучного інтелекту для покращення аналітики даних, оновлений інтерфейс користувача з налаштовуваними панелями, а також підвищену продуктивність та безпеку. Вдосконалені API забезпечують інтеграцію з різними сторонніми сервісами.

SOURCES: abc_release_notes.pdf, abc_user_manual_ua.docx

QUESTION: {question}
LONG QUESTION: {long_question}
=========
{summaries}
=========
FINAL ANSWER:"""

RETRIEVAL_CHAIN_PROMPT = PromptTemplate(
    template=RETRIEVAL_CHAIN_PROMPT_TEMPLATE,
    input_variables=["summaries", "question", "long_question"],
)


class DocumentRetrievalChain(RetrievalQAWithSourcesChain):
    @classmethod
    def from_chain_type(
        cls,
        *args,
        chain_type_kwargs={"prompt": RETRIEVAL_CHAIN_PROMPT},
        **kwargs
    ):
        return super().from_chain_type(
            *args,
            chain_type_kwargs=chain_type_kwargs,
            **kwargs
        )

    def invoke(self, query):

        # Parse input string from the agent
        try:
            parsed_query = ast.literal_eval(query)
        except:
            parsed_query = query

        # Extract dictionary values from parsed string
        if isinstance(parsed_query, dict):
            question = parsed_query.get("question", "")
            long_question = parsed_query.get("long_question", "")
        else:
            question = query
            long_question = question

        chain_output = super().invoke({"question": question, "long_question": long_question})

        output = {"answer": chain_output["answer"], "sources": chain_output["sources"]}

        return output