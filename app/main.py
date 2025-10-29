from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from app.chatbot import DE5ChatAssistant

# Initialize FastAPI app
app = FastAPI(
    title="DE5 Chat Assistant API",
    description="Backend API for DE5 Chat Assistant with GPT-4 integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chat assistant
chat_assistant = None

def get_chat_assistant():
    """Lazy initialization of chat assistant"""
    global chat_assistant
    if chat_assistant is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )
        chat_assistant = DE5ChatAssistant(api_key=api_key)
    return chat_assistant

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is machine learning?"
            }
        }

class ChatResponse(BaseModel):
    response: str
    context_used: bool = False  # Hook for future RAG integration
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Machine learning is...",
                "context_used": False
            }
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "ok"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to DE5 Chat Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Chat endpoint - Main functionality
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint that processes user messages and returns AI-generated responses.
    
    Currently integrates with GPT-4 via OpenAI API.
    Future enhancement: Will support document/context RAG integration.
    
    Args:
        request: ChatRequest containing the user's message
    
    Returns:
        ChatResponse with the AI-generated response
    
    Raises:
        HTTPException: If the assistant is not properly configured or if processing fails
    """
    try:
        assistant = get_chat_assistant()
        
        # Generate response using GPT-4
        ai_response = assistant.generate_response(request.message)
        
        # Hook for future RAG integration:
        # context = await retrieve_relevant_context(request.message)
        # ai_response = assistant.generate_response_with_context(
        #     message=request.message,
        #     context=context
        # )
        
        return ChatResponse(
            response=ai_response,
            context_used=False  # Will be True when RAG is integrated
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
