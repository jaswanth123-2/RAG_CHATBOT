from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
   
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = []
        start = 0
       
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
           
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
               
                if break_point > self.chunk_size * 0.5:
                    end = start + break_point + 1
                    chunk_text = text[start:end]
           
            chunk = {
                'text': chunk_text.strip(),
                'metadata': {
                    **metadata,
                    'chunk_id': len(chunks),
                    'start_char': start,
                    'end_char': end
                }
            }
           
            chunks.append(chunk)
            start = end - self.chunk_overlap
       
        return chunks
   
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []
       
        for doc in documents:
            chunks = self.chunk_text(doc['text'], doc['metadata'])
            all_chunks.extend(chunks)
       
        return all_chunks

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded! Dimension: {dim}")
   
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)
   
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
   
    def embed_chunks(self, chunks: List[Dict], show_progress: bool = True) -> List[Dict]:
        texts = [chunk['text'] for chunk in chunks]
       
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_texts(texts, show_progress)
       
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
       
        print(f"✓ Embeddings generated!")
        return chunks

if __name__ == "__main__":
    sample_text = "This is a test. " * 100
   
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text, {'source': 'test.txt'})
   
    print(f"Created {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['text'][:100]}...")