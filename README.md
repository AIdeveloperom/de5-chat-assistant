# de5-chat-assistant
AI-powered, blockchain-integrated chat assistant for DE5.tech

## Tech Stack

This project is built with the following technologies:

- **FastAPI Backend**: Modern, fast web framework for building APIs with Python
- **OpenAI GPT-4 Integration**: Advanced AI language model for intelligent chat responses
- **SQLAlchemy Database**: SQL toolkit and Object-Relational Mapping (ORM) for Python
- **Pinecone Vector Database** (Optional): Vector database for semantic search and similarity matching

## Getting Started

Follow these instructions to get the backend running locally on your machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AIdeveloperom/de5-chat-assistant.git
cd de5-chat-assistant
```

2. Install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables (create a `.env` file):
```bash
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key  # Optional
DATABASE_URL=your_database_url
```

### Running the Backend

Start the FastAPI server using uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

You can access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### POST /chat

The main chat endpoint that processes user messages and returns AI-generated responses.

**Endpoint**: `POST /chat`

**Request Body**:
```json
{
  "message": "What is machine learning?"
}
```

**Response**:
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "context_used": false
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is machine learning?"}'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "What is machine learning?"}
)

data = response.json()
print(data["response"])
```

**JavaScript/Fetch Example**:
```javascript
fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'What is machine learning?'
  })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

### GET /health

Health check endpoint to verify the API is running.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok"
}
```

### GET /

Root endpoint with API information.

**Response**:
```json
{
  "message": "Welcome to DE5 Chat Assistant API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

## Features

### Current Implementation

- **GPT-4 Integration**: The `/chat` endpoint currently uses OpenAI's GPT-4 model for generating intelligent responses
- **Pydantic Models**: Request/response validation using Pydantic models
- **Error Handling**: Comprehensive error handling with meaningful error messages
- **CORS Support**: Configured for cross-origin requests (configure appropriately for production)
- **Interactive Documentation**: Auto-generated API docs available at `/docs`

### Future Enhancements (RAG Integration)

The codebase includes hooks for future document/context RAG (Retrieval-Augmented Generation) integration:

```python
# Future RAG integration hook (currently commented out in main.py):
# context = await retrieve_relevant_context(request.message)
# ai_response = assistant.generate_response_with_context(
#     message=request.message,
#     context=context
# )
```

When RAG is integrated:
- The `context_used` field in responses will be set to `True`
- Relevant documents/context will be retrieved from a vector database
- Responses will be augmented with domain-specific knowledge

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key for GPT-4 access |
| `PINECONE_API_KEY` | No | Pinecone API key for vector database (future RAG integration) |
| `DATABASE_URL` | No | Database connection string for persistence |

## Project Structure

```
de5-chat-assistant/
├── app/
│   ├── chatbot.py        # DE5ChatAssistant class
│   └── main.py           # FastAPI application and endpoints
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .env                 # Environment variables (not in repo)
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful response
- `500`: Internal server error (e.g., missing API key, processing error)

Error response format:
```json
{
  "detail": "Error processing chat request: <error message>"
}
```

## License

MIT License - see LICENSE file for details
