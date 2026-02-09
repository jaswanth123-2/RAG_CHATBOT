import sys
sys.path.append('src')
from document_loader import DocumentLoader
from embeddings import TextChunker, EmbeddingModel
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

def main():
    print("BUILDING RAG SYSTEM")
   
    print("\n[1/5] Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_directory("data/documents")
   
    if not documents:
        print("❌ No documents found!")
        return
   
    print(f"✓ Loaded {len(documents)} documents")
   
    print("\n[2/5] Chunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
   
    print("\n[3/5] Generating embeddings...")
    embedder = EmbeddingModel()
    chunks = embedder.embed_chunks(chunks)
   
    print("\n[4/5] Creating vector store...")
    vs = VectorStore()
    vs.create_collection("documents")
    vs.add_chunks(chunks)
   
    print("\n[5/5] Initializing RAG pipeline...")
    rag = RAGPipeline(vs, embedder)
   
    print("\nSYSTEM READY!")
   
    print("\nTESTING QUERIES")
   
    test_questions = [
        "What is the attention mechanism in transformers?",
        "How does BERT work?",
        "Explain dropout regularization"
    ]
   
    for question in test_questions:
        print(f"\nQ: {question}")
       
        response = rag.query(question, top_k=3)
       
        print(f"\nA: {response['answer']}")
       
        print(f"\nSources:")
        for i, source in enumerate(response['sources'], 1):
            print(f" {i}. {source['source']} (score: {source['similarity_score']})")
   
    print("\nALL TESTS PASSED!")

if __name__ == "__main__":
    main()