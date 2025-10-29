import os
import json
from typing import List, Dict, Optional, Any
import requests
from bs4 import BeautifulSoup
import numpy as np
from openai import OpenAI
import faiss


class DE5ChatAssistant:
    """
    DE5 Chat Assistant for answering questions about DE5 protocol using RAG.
    Supports loading documents from whitepapers, web content, and public docs,
    embedding texts for semantic search, and generating AI responses using GPT-4.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DE5ChatAssistant.
        
        Args:
            api_key: OpenAI API key. If not provided, will use OPENAI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.embedding_dimension = 1536  # OpenAI embedding dimension

    def load_whitepaper(self, file_path: str) -> Dict[str, Any]:
        """
        Load whitepaper content from a file.
        
        Args:
            file_path: Path to the whitepaper file (txt, md, or json)
        
        Returns:
            Dictionary containing the loaded document metadata and content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    content = json.load(f)
                    if isinstance(content, dict):
                        text = content.get('content', str(content))
                    else:
                        text = str(content)
                else:
                    text = f.read()
            
            doc = {
                'content': text,
                'source': file_path,
                'type': 'whitepaper'
            }
            self.documents.append(doc)
            return doc
        except Exception as e:
            raise Exception(f"Error loading whitepaper from {file_path}: {str(e)}")

    def load_web_content(self, url: str) -> Dict[str, Any]:
        """
        Load content from a web URL.
        
        Args:
            url: Web URL to scrape content from
        
        Returns:
            Dictionary containing the loaded document metadata and content
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            doc = {
                'content': text,
                'source': url,
                'type': 'web_content'
            }
            self.documents.append(doc)
            return doc
        except Exception as e:
            raise Exception(f"Error loading web content from {url}: {str(e)}")

    def load_public_docs(self, docs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Load public documents from a list of dictionaries.
        
        Args:
            docs: List of dictionaries with 'content' and optional 'metadata' keys
        
        Returns:
            List of loaded document dictionaries
        """
        loaded_docs = []
        for i, doc in enumerate(docs):
            doc_data = {
                'content': doc.get('content', ''),
                'source': doc.get('source', f'public_doc_{i}'),
                'type': 'public_doc',
                'metadata': doc.get('metadata', {})
            }
            self.documents.append(doc_data)
            loaded_docs.append(doc_data)
        
        return loaded_docs

    def chunk_documents(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for better embedding and retrieval.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        
        Returns:
            List of document chunks
        """
        chunks = []
        for doc in self.documents:
            content = doc['content']
            start = 0
            
            while start < len(content):
                end = start + chunk_size
                chunk_text = content[start:end]
                
                chunks.append({
                    'content': chunk_text,
                    'source': doc['source'],
                    'type': doc['type'],
                    'chunk_id': len(chunks)
                })
                
                start += chunk_size - overlap
        
        return chunks

    def embed_texts(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate embeddings for texts using OpenAI's embedding model.
        
        Args:
            texts: List of texts to embed. If None, uses chunked documents.
        
        Returns:
            NumPy array of embeddings
        """
        if texts is None:
            chunks = self.chunk_documents()
            texts = [chunk['content'] for chunk in chunks]
            self.document_chunks = chunks
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings).astype('float32')
        return self.embeddings

    def build_search_index(self):
        """
        Build FAISS index for semantic search.
        """
        if self.embeddings is None:
            self.embed_texts()
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index.add(self.embeddings)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search to find relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of relevant document chunks with scores
        """
        if self.index is None:
            self.build_search_index()
        
        # Embed the query
        response = self.client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.document_chunks):
                result = self.document_chunks[idx].copy()
                result['relevance_score'] = float(1 / (1 + dist))  # Convert distance to similarity score
                result['rank'] = i + 1
                results.append(result)
        
        return results

    def generate_response(self, query: str, use_rag: bool = True, model: str = "gpt-4") -> str:
        """
        Generate AI response using GPT-4 with or without RAG.
        
        Args:
            query: User query
            use_rag: Whether to use Retrieval Augmented Generation
            model: OpenAI model to use (default: gpt-4)
        
        Returns:
            Generated response string
        """
        if use_rag and self.documents:
            # Retrieve relevant context
            relevant_docs = self.semantic_search(query, top_k=5)
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Source: {doc['source']}\n{doc['content']}"
                for doc in relevant_docs
            ])
            
            # Create prompt with context
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in the DE5 protocol. "
                               "Use the provided context to answer questions accurately. "
                               "If the context doesn't contain relevant information, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        else:
            # Generate response without RAG
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in the DE5 protocol."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        
        # Generate response
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content

    def chat(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Main chat interface that combines search and response generation.
        
        Args:
            query: User query
            use_rag: Whether to use RAG
        
        Returns:
            Dictionary containing response and metadata
        """
        if use_rag and self.documents:
            relevant_docs = self.semantic_search(query)
            response = self.generate_response(query, use_rag=True)
            
            return {
                'query': query,
                'response': response,
                'sources': [doc['source'] for doc in relevant_docs],
                'relevant_chunks': relevant_docs,
                'method': 'RAG'
            }
        else:
            response = self.generate_response(query, use_rag=False)
            return {
                'query': query,
                'response': response,
                'method': 'Direct GPT-4'
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded documents and index.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.document_chunks) if hasattr(self, 'document_chunks') else 0,
            'index_built': self.index is not None,
            'embedding_dimension': self.embedding_dimension
        }
