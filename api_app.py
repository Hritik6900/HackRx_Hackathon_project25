import sys
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
from langchain.vectorstores import FAISS
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
    webhook_url: Optional[str] = None

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
    """Enhanced: Setup QA system with FAISS (ChromaDB replacement)"""
    try:
        text_chunks = get_text_chunks(pdf_text)
        
        if not text_chunks:
            raise ValueError("No text chunks generated from PDF")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # FAISS instead of ChromaDB - No build issues!
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
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
        
        return conversation_chain, None, vectorstore
        
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

@app.get("/debug")
async def debug_info():
    try:
        import langchain
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        import google.generativeai as genai
        
        # Test Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Test message")
        
        # Test embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Test PDF access
        pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        pdf_response = requests.get(pdf_url, timeout=10)
        
        return {
            "status": "debug_success",
            "langchain_version": langchain._version_,
            "google_api_key_exists": bool(api_key),
            "google_api_test": test_response.text[:100] if test_response.text else "No response",
            "embeddings_test": "Success",
            "pdf_access_status": pdf_response.status_code,
            "pdf_size": len(pdf_response.content),
            "python_version": sys.version[:50]
        }
    except Exception as e:
        return {
            "status": "debug_failed", 
            "error": str(e),
            "error_type": type(e)._name_,
            "google_api_key_exists": bool(os.getenv("GOOGLE_API_KEY"))
        }

@app.post("/hackrx/run")
async def process_questions(
    request: QuestionRequest,
    token: str = Depends(verify_token)
):
    """Speed-optimized processing - returns simple answers array"""
    try:
        # Validation
        if not request.documents:
            raise HTTPException(400, "Documents URL is required")
        
        if not request.questions:
            raise HTTPException(400, "At least one question is required")
        
        # Process all questions (no limit for hackathon)
        questions = request.questions
        
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
                            "answer": raw_answer[:500] if raw_answer else "Unable to generate answer",
                            "matched_clauses": matched_clauses,
                            "rationale": "Generated from document analysis",
                            "confidence": 0.75
                        })
                        
                except Exception as e:
                    answers.append({
                        "answer": f"Unable to process this question due to processing constraints",
                        "matched_clauses": [],
                        "rationale": "Error occurred during processing",
                        "confidence": 0.0
                    })
            
            return answers
        
        # Execute with timeout
        answers = await asyncio.wait_for(fast_process(), timeout=25.0)
        
        # Extract ONLY answer strings for platform compatibility
        simple_answers = []
        for ans in answers:
            if isinstance(ans, dict) and "answer" in ans:
                answer_text = str(ans["answer"]).strip()
                # Ensure meaningful answer
                if answer_text and len(answer_text) > 0:
                    simple_answers.append(answer_text)
                else:
                    simple_answers.append("Unable to extract relevant information from the document")
            else:
                simple_answers.append("Processing failed")
        
        # Send webhook if provided (with simple format)
        if request.webhook_url:
            webhook_data = {"answers": simple_answers}
            await send_webhook(request.webhook_url, webhook_data)
        
        # Return EXACTLY what platform expects - only answers array
        return {"answers": simple_answers}
        
    except asyncio.TimeoutError:
        # Even timeout should return proper format
        return {"answers": ["Request timeout - unable to process questions within time limit"]}
    except HTTPException:
        raise
    except Exception as e:
        # Even errors should return proper format
        return {"answers": [f"Processing error occurred: {str(e)}"]}

@app.post("/webhook/callback")
async def webhook_callback(request: Request):
    """Enhanced: Receive webhook callbacks from external systems"""
    try:
        data = await request.json()
        print("Webhook received:", json.dumps(data, indent=2))
        
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
            "Platform-compatible simple response format",
            "Optimized PDF processing",
            "AI-powered question answering",
            "Vector-based document search",
            "Robust error handling"
        ],
        "endpoints": ["/hackrx/run", "/webhook/callback", "/health"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "hackrx-pdf-qa-api",
        "format": "platform-compatible"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
