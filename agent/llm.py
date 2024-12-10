import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLM:
    def __init__(self):
        self.model = self._create_llm()
        
    def _create_llm(self):
        model = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        return model