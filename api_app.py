from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import json
import asyncio
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import io
import tempfile
import shutil


load_dotenv()

app = FastAPI(title="Enhanced PDF QA API", description="Hackathon PDF Question Answering API with Advanced Features")
security = HTTPBearer()

API_KEY = os.getenv("API_KEY", "hackrx-api-key-2025")

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]
    webhook_url: Optional[str] = None  # Enhanced: Optional webhook support

class EnhancedAnswer(BaseModel):
    answer: str
    matched_clauses: List[dict]
    rationale: str
    confidence: float

class QuestionResponse(BaseModel):
    success: bool = True
    status: str = "completed"
    processing_time: Optional[float] = None
    answers: List[dict]
    metadata: Optional[dict] = None


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def extract_answer(response) -> str:
    """Enhanced: Robust answer extraction from LangChain output"""
    if isinstance(response, dict):
        if "answer" in response:
            return response["answer"]
        elif "result" in response:
            return response["result"]
        elif "output" in response:
            return response["output"]
        # Handle chat_history fallback
        elif "chat_history" in response and response["chat_history"]:
            for msg in reversed(response["chat_history"]):
                if hasattr(msg, "type") and msg.type == "ai":
                    return msg.content
        # Handle source_documents for additional context
        elif "source_documents" in response:
            return "Answer extracted from source documents."
    elif isinstance(response, str):
        return response
    return "No answer found."

def safe_json_parse(model_output: str) -> dict:
    """Enhanced: Safe JSON parsing with fallback"""
    try:
        # Find first valid JSON substring
        first_brace = model_output.find("{")
        if first_brace != -1:
            json_str = model_output[first_brace:]
            return json.loads(json_str)
    except Exception:
        pass
    return None

def create_enhanced_prompt(question: str, context: str) -> str:
    """Enhanced: Create structured prompt for JSON output"""
    return f"""
Using ONLY the following policy/contract clauses:

{context}

Answer the question: "{question}"

Respond ONLY in the following JSON format:
{{
  "answer": "Direct answer to the question",
  "matched_clauses": [
     {{"clause_number": "Section X.Y", "text": "Relevant clause text"}}
  ],
  "rationale": "Explanation of how the clauses support the answer",
  "confidence": 0.95
}}

If you cannot find specific information, be honest and provide a lower confidence score.
"""

def download_pdf_optimized(url: str) -> str:
    """Ultra-fast PDF download with streaming"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # Stream download for faster processing
        with requests.get(url, headers=headers, timeout=20, stream=True, verify=False) as response:
            response.raise_for_status()
            
            # Read in chunks for memory efficiency
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 10 * 1024 * 1024:  # Limit to 10MB
                    break
        
        pdf_reader = PdfReader(io.BytesIO(content))
        
        # Process only first 10 pages for speed
        text = ""
        for page in pdf_reader.pages[:10]:
            text += page.extract_text() + "\n"
            if len(text) > 50000:  # Limit text length
                break
        
        return text
    except Exception as e:
        raise HTTPException(400, f"PDF error: {str(e)}")


def get_text_chunks(text: str):
    """Split text into chunks with metadata"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def setup_enhanced_qa_system(pdf_text: str):
    """Enhanced: Setup QA system with better retrieval"""
    try:
        text_chunks = get_text_chunks(pdf_text)
        
        if not text_chunks:
            raise ValueError("No text chunks generated from PDF")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        temp_dir = tempfile.mkdtemp()
        
        vectorstore = Chroma.from_texts(
            texts=text_chunks, 
            embedding=embeddings,
            persist_directory=temp_dir
        )
        
        # Enhanced: More precise retrieval
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,  # Lower for more consistent JSON
            max_output_tokens=1024,  # Higher for detailed responses
            convert_system_message_to_human=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # More context
            memory=None,
            return_source_documents=True  # Get source docs for clauses
        )
        
        return conversation_chain, temp_dir, vectorstore
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting up QA system: {str(e)}"
        )

async def send_webhook(webhook_url: str, data: dict):
    """Enhanced: Send results to webhook URL"""
    try:
        response = requests.post(
            webhook_url, 
            json=data, 
            timeout=15,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Webhook delivery failed: {e}")
        return False

@app.post("/hackrx/run", response_model=QuestionResponse)
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Speed-optimized processing with status tracking"""
    start_time = time.time()
    temp_dir = None
    
    try:
        # Validation
        if not request.documents:
            raise HTTPException(400, "Documents URL is required")
        
        if not request.questions:
            raise HTTPException(400, "At least one question is required")
        
        # Limit questions for speed
        questions = request.questions[:3]  # Max 3 questions
        
        # Fast processing with timeout
        async def fast_process():
            # Download and process PDF
            pdf_text = download_pdf_optimized(request.documents)
            if not pdf_text.strip():
                raise HTTPException(400, "No text content found in PDF")
            
            # Setup QA system
            qa_system, temp_dir, vectorstore = setup_enhanced_qa_system(pdf_text)
            
            # Process questions
            answers = []
            for question in questions:
                if not question.strip():
                    answers.append({
                        "answer": "Invalid or empty question provided.",
                        "matched_clauses": [],
                        "rationale": "Question was empty or invalid",
                        "confidence": 0.0
                    })
                    continue
                    
                try:
                    # Get relevant context
                    relevant_docs = vectorstore.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Create enhanced prompt
                    enhanced_prompt = create_enhanced_prompt(question, context)
                    
                    # Get LLM response
                    response = qa_system({'question': enhanced_prompt})
                    raw_answer = extract_answer(response)
                    
                    # Try to parse JSON response
                    parsed_json = safe_json_parse(raw_answer)
                    
                    if parsed_json and "answer" in parsed_json:
                        answers.append({
                            "answer": parsed_json["answer"],
                            "matched_clauses": parsed_json.get("matched_clauses", []),
                            "rationale": parsed_json.get("rationale", "AI-generated response"),
                            "confidence": float(parsed_json.get("confidence", 0.75))
                        })
                    else:
                        # Fallback response
                        matched_clauses = []
                        if hasattr(response, 'source_documents') and response.source_documents:
                            for i, doc in enumerate(response.source_documents[:3]):
                                matched_clauses.append({
                                    "clause_number": f"Section {i+1}",
                                    "text": doc.page_content[:200] + "..."
                                })
                        
                        answers.append({
                            "answer": raw_answer[:500],
                            "matched_clauses": matched_clauses,
                            "rationale": "Generated from document analysis",
                            "confidence": 0.75
                        })
                        
                except Exception as e:
                    answers.append({
                        "answer": f"Error processing question: {str(e)}",
                        "matched_clauses": [],
                        "rationale": "Error occurred during processing",
                        "confidence": 0.0
                    })
            
            return answers
        
        # Execute with timeout
        answers = await asyncio.wait_for(fast_process(), timeout=25.0)
        processing_time = time.time() - start_time
        
        # Create response with status
        final_response = QuestionResponse(
            success=True,
            status="completed",
            processing_time=round(processing_time, 2),
            answers=answers,
            metadata={
                "questions_processed": len(answers),
                "document_url": request.documents,
                "timestamp": time.time()
            }
        )
        
        # Send webhook if provided
        if request.webhook_url:
            webhook_data = final_response.dict()
            await send_webhook(request.webhook_url, webhook_data)
        
        return final_response
        
    except asyncio.TimeoutError:
        processing_time = time.time() - start_time
        return QuestionResponse(
            success=False,
            status="timeout",
            processing_time=round(processing_time, 2),
            answers=[],
            metadata={"error": "Request timeout - try fewer questions"}
        )
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        return QuestionResponse(
            success=False,
            status="error",
            processing_time=round(processing_time, 2),
            answers=[],
            metadata={"error": str(e)}
        )
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


@app.post("/webhook/callback")
async def webhook_callback(request: Request):
    """Enhanced: Receive webhook callbacks from external systems"""
    try:
        data = await request.json()
        print("Webhook received:", json.dumps(data, indent=2))
        
        # You can add custom processing logic here
        # For example: store in database, trigger other processes, etc.
        
        return {
            "status": "received",
            "detail": "Webhook processed successfully",
            "timestamp": data.get("timestamp", "not provided"),
            "processed": True
        }
    except Exception as e:
        return {
            "status": "error",
            "detail": f"Failed to process webhook: {str(e)}"
        }

@app.get("/")
async def root():
    return {
        "message": "Enhanced Hackathon PDF QA API is running",
        "version": "2.0.0",
        "features": [
            "Structured JSON responses",
            "Clause extraction and citation",
            "Confidence scoring",
            "Webhook support",
            "Robust answer extraction"
        ],
        "endpoints": ["/hackrx/run", "/webhook/callback", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "enhanced-pdf-qa-api",
        "features_active": ["pdf_processing", "ai_qa", "webhooks", "structured_output"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
