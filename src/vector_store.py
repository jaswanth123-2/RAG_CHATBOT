import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import os

class VectorStore:
    def __init__(self, persist_directory: str = "vectordb"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        print(f"✓ Vector store initialized: {persist_directory}")
    
    def create_collection(self, collection_name: str = "documents"):
        try:
            try:
                self.client.delete_collection(collection_name)
                print(f"Deleted old collection")
            except:
                pass
            
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Created collection: {collection_name}")
            
        except Exception as e:
            print(f"Getting existing collection...")
            self.collection = self.client.get_collection(collection_name)
    
    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks...")
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        documents = [chunk['text'] for chunk in chunks]
        embeddings = [chunk['embedding'].tolist() for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            meta = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float)):
                    meta[key] = value
                else:
                    meta[key] = str(value)
            metadatas.append(meta)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end]
            )
            
            print(f"  Added batch {i//batch_size + 1}")
        
        print(f"✓ Added {len(chunks)} chunks")
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        matches = []
        for i in range(len(results['ids'][0])):
            match = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],
                'id': results['ids'][0][i]
            }
            matches.append(match)
        
        return matches
    
    def get_stats(self) -> Dict:
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection.name
        }