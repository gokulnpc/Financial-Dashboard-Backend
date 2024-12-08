from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Optional
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from reportlab.pdfgen import canvas
from io import BytesIO
import os
from dotenv import load_dotenv
import base64
from typing import List, Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import logging
import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from utils import generate_dashboard, generate_report_content, classify_document
from openai import OpenAI

load_dotenv()
app = FastAPI()

class LargeFileMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # You can add custom file handling logic here if needed
        response = await call_next(request)
        return response
    
# Add middleware configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=3600
)

app.add_middleware(LargeFileMiddleware)

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
index_name = "financial-docs"

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

class ChatRequest(BaseModel):
    question: str
    chat_history: List[tuple] = []
    namespace: str

class ReportRequest(BaseModel):
    topics: List[str]
    namespace: str

class DiagnosticRequest(BaseModel):
    namespace: str
     
# Initialize vectorstore with namespace support
def get_vectorstore(namespace: str):
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    
@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    namespace: str = Header(..., description="Namespace identifier for document isolation")
):
    try:
        logger.info(f"Starting upload process for {len(files)} files in namespace: {namespace}")
        documents = []
        doc_classifications = []

        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory: {temp_dir}")

            for file in files:
                try:
                    logger.info(f"Processing file: {file.filename}")
                    
                    if not file.filename.endswith('.pdf'):
                        logger.warning(f"Invalid file type: {file.filename}")
                        raise HTTPException(status_code=400, detail="Only PDF files are supported")

                    # Create temporary file path
                    temp_path = os.path.join(temp_dir, f"temp_{file.filename}")
                    logger.info(f"Temporary file path: {temp_path}")

                    # Read and write in chunks to handle large files
                    content = await file.read()
                    logger.info(f"Read file content: {len(content)} bytes")

                    try:
                        with open(temp_path, "wb") as f:
                            f.write(content)
                        logger.info(f"Wrote content to temporary file: {temp_path}")

                        # Load PDF
                        logger.info("Loading PDF with PyPDFLoader")
                        loader = PyPDFLoader(temp_path)
                        docs = loader.load()
                        logger.info(f"Loaded {len(docs)} pages from PDF")
                        documents.extend(docs)
                    except Exception as file_error:
                        logger.error(f"Error processing file: {str(file_error)}")
                        logger.error(traceback.format_exc())
                        raise HTTPException(
                            status_code=500,
                            detail=f"Error processing file {file.filename}: {str(file_error)}"
                        )
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing file {file.filename}: {str(e)}"
                    )

            logger.info("Starting document splitting")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(docs)} chunks")

            logger.info("Creating vector store")
            try:
                vectorstore = PineconeVectorStore.from_documents(
                    docs,
                    embedding=embeddings,
                    index_name=index_name,
                    namespace=namespace
                )
                logger.info(f"Vector store created successfully for namespace: {namespace}")
            except Exception as vec_error:
                logger.error(f"Vector store error: {str(vec_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating vector store: {str(vec_error)}"
                )

        logger.info("Upload completed successfully")
        return {
            "message": "Documents uploaded successfully",
            "documents": doc_classifications
        }

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
   
@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    try:
        # Retrieve documents
        print(f"Retrieving documents for topics: {request.topics}")
        print(f"Namespace: {request.namespace}")
        vectorstore = get_vectorstore(request.namespace)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
        documents = []
        
        for topic in request.topics:
            docs = retriever.get_relevant_documents(topic)
            documents.extend(docs)
        print(f"Documents retrieved for namespace {request.namespace}", documents)
        # Generate report content
        report_content = generate_report_content(request.topics, documents, llm)
        
        # Create PDF
        buffer = BytesIO()
        p = canvas.Canvas(buffer)
        y_position = 750
        
        # Add content with proper formatting
        for line in report_content.split('\n'):
            # Reset page if needed
            if y_position < 50:
                p.showPage()
                y_position = 750
                
            # Format headers
            if line.startswith('#'):
                p.setFont("Helvetica-Bold", 14)
                # Remove # symbol from the header
                line = line.lstrip('#').strip()
                y_position -= 30
            else:
                p.setFont("Helvetica", 12)
                y_position -= 20
            
            # Handle long lines with text wrapping
            text = p.beginText(50, y_position)
            text.setFont("Helvetica", 12)
            
            # Wrap text at 80 characters
            words = line.split()
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= 80:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    text.textLine(' '.join(current_line))
                    y_position -= 15
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                text.textLine(' '.join(current_line))
            
            p.drawText(text)
        
        p.save()
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        
        # Return both the content and PDF as base64
        return {
            "content": report_content,
            "pdf_base64": base64.b64encode(pdf_bytes).decode('utf-8')
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )

@app.post("/generate-dashboard")
async def dashboard_endpoint(
    request: Dict[str, List[str]],
    namespace: str = Header(..., description="Namespace identifier for document isolation")
):
    print(f"Request: {request}")
    print(f"Namespace: {namespace}")
    try:
        result = await generate_dashboard(request, namespace)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating dashboard: {str(e)}"
        )

@app.get("/info")
async def info():
    return {
        "message": "Welcome to the Financial Document Analysis API. Use the /upload endpoint to upload financial documents, /chat to chat with the documents, /generate-report to generate a detailed financial report, and /generate-dashboard to generate a financial dashboard."
    }
    
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300,
        limit_concurrency=20,
        limit_max_requests=20,
        backlog=200,
    )