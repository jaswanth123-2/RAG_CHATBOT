from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class RAGPipeline:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
       
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
       
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
       
        print(f"✓ RAG Pipeline initialized")
   
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.embed_text(query)
        results = self.vector_store.search(query_embedding.tolist(), top_k)
        return results
   
    def format_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            text = chunk['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")
       
        context = "\n\n".join(context_parts)
       
        prompt = f"""You are a helpful AI assistant answering questions about machine learning research papers.


Context from research papers:
{context}

Question: {query}

Instructions:
- Explain clearly and concisely in your own words — do NOT copy sentences directly unless you are quoting
- Use simple language suitable for intermediate learners
- Structure your answer: 1. Definition, 2. How it works, 3. Why it's effective
- Cite sources using [Source X] when you use their ideas
- If information is missing or unclear, say so honestly
- End with a short practical tip if relevant

Answer:"""      
        return prompt
   
    def generate_answer(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant for ML research papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
           
            return completion.choices[0].message.content
       
        except Exception as e:
            return f"Error: {str(e)}"
   
    def query(self, question: str, top_k: int = 3) -> Dict:
        print(f"\n🔍 Query: {question}")
       
        context_chunks = self.retrieve_context(question, top_k)
        print(f"✓ Retrieved {len(context_chunks)} chunks")
       
        prompt = self.format_prompt(question, context_chunks)
        answer = self.generate_answer(prompt)
        print(f"✓ Answer generated")
       
        sources = []
        for chunk in context_chunks:
            sources.append({
                'source': chunk['metadata'].get('source', 'Unknown'),
                'text': chunk['text'][:200] + "...",
                'similarity_score': round(chunk['score'], 3)
            })
       
        return {
            'question': question,
            'answer': answer,
            'sources': sources
        }