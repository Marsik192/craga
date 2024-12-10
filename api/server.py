from fastapi import HTTPException, FastAPI, status
import os
from pydantic import BaseModel
import base64
from io import BytesIO

from agent.agent import Agent
from api.exception_handling import handle_exceptions

from data_pipeline.vectorstore import VectorStore
from data_pipeline.documents_preparation import load_single_document

agent = Agent()

app = FastAPI()

handle_exceptions(
    app=app,
    on_error=agent.interrupt_generation
)

class PDFRequest(BaseModel):
    files: list[str]  # List of base64-encoded file bytes

class AgentInput(BaseModel):
    input: str


@app.get("/healthcheck")
def healthcheck():
    """
    A public endpoint that shows if the API is running
    """
    return {"description": "Agent API is up and running..."}

@app.post("/get_chat_completion")
async def get_chat_completion(
    request: AgentInput
):
    """
    An endpoint that returns a chat completion.

    Query parameters:
        input: The input to the agent.

    Returns:
        HTTP response containing the user input and agent output.
    """

    # Raise an error if the input is empty
    if len(request.input) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input cannot be empty",
        )

    llm_response = await agent.generate_response(request.input)

    response = {
        "input": llm_response["input"],
        "output": llm_response["output"]
    }

    return response

@app.post("/ingest_data")
async def ingest_data(
    request: PDFRequest
):
    """
    An endpoint that accepts .zip from user.
    """
    # Validate input is not empty
    if not request.files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="No files provided in the request."
        )

    pdf_files = []

    for i, file_data in enumerate(request.files):
        try:
            # Validate Base64
            try:
                pdf_bytes = base64.b64decode(file_data)
            except base64.binascii.Error:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail=f"File {i} is not a valid Base64 string."
                )

            # Check if PDF content can be processed
            pdf_stream = BytesIO(pdf_bytes)

            pdf_files.append(pdf_stream)

        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"An unexpected error occurred while processing file {i}: {str(e)}"
            )
    
    all_documents = []
    for file in pdf_files:
        documents = load_single_document(file)
        all_documents += documents

    chroma_client = VectorStore("vectorstore_data")
    chroma_client.clear_vectorstore()
    vectorstore = chroma_client.create_vectorstore(all_documents)

    agent.update_vectorstore(vectorstore)

    return {"description": "The PDFs were processed successfully"}


# Not necessary but useful
@app.post("/interrupt")
def interrupt_completion():
    """
    An endpoint that interrupts the agent's generation.
    """
    agent.interrupt_generation()
    response = {"description": "Request interrupted"}

    return response

# Not necessary but useful
@app.post("/clean_history")
def clean_history():
    """
    An endpoint that cleans the chat history of the agent.
    """
    agent.clean_memory()
    response = {"description": "History cleaned"}

    return response