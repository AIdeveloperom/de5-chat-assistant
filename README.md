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
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
