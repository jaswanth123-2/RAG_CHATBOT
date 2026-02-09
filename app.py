import streamlit as st
import os
import sys
sys.path.append('src')
from document_loader import DocumentLoader
from embeddings import TextChunker, EmbeddingModel
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def initialize_system():
    with st.spinner("Loading system..."):
        loader = DocumentLoader()
        chunker = TextChunker(chunk_size=500, chunk_overlap=50)
        embedder = EmbeddingModel()
        vector_store = VectorStore()
       
        try:
            vector_store.collection = vector_store.client.get_collection("documents")
            st.success("✓ Loaded existing database")
        except:
            st.info("No database found. Click 'Load Documents' below.")
            vector_store.create_collection("documents")
       
        rag = RAGPipeline(vector_store, embedder)
       
    return loader, chunker, embedder, vector_store, rag

def load_documents(loader, chunker, embedder, vector_store):
    with st.spinner("Loading documents..."):
        documents = loader.load_directory("data/documents")
        if not documents:
            st.error("No documents found in data/documents/")
            return False
       
        chunks = chunker.chunk_documents(documents)
        st.info(f"Created {len(chunks)} chunks")
       
        chunks = embedder.embed_chunks(chunks, show_progress=False)
       
        vector_store.add_chunks(chunks)
       
        st.success(f"✓ Loaded {len(documents)} documents!")
        return True

def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Ask questions about ML research papers")
   
    if 'messages' not in st.session_state:
        st.session_state.messages = []
   
    loader, chunker, embedder, vector_store, rag = initialize_system()
   
    with st.sidebar:
        st.header("📚 Documents")
       
        if st.button("📥 Load Documents", use_container_width=True):
            load_documents(loader, chunker, embedder, vector_store)
       
        st.divider()
       
        try:
            stats = vector_store.get_stats()
            st.metric("Total Chunks", stats['total_chunks'])
        except:
            st.info("No documents loaded")
       
        st.divider()
       
        top_k = st.slider("Retrieved Chunks", 1, 10, 3)
       
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
   
    st.header("💬 Chat")
   
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
           
            if "sources" in msg:
                with st.expander("📚 Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {src['source']}** (score: {src['similarity_score']})")
                        st.text(src['text'])
   
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
       
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.query(prompt, top_k=top_k)
               
                st.markdown(response['answer'])
               
                with st.expander("📚 Sources"):
                    for i, src in enumerate(response['sources'], 1):
                        st.markdown(f"**{i}. {src['source']}** (score: {src['similarity_score']})")
                        st.text(src['text'])
       
        st.session_state.messages.append({
            "role": "assistant",
            "content": response['answer'],
            "sources": response['sources']
        })

if __name__ == "__main__":
    main()