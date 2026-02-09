RAG Chatbot - AI-Powered Research Paper Q&A

Ask questions about machine learning research papers using Retrieval-Augmented Generation (RAG)

Live Demo

HuggingFace Spaces: [DEMO](https://huggingface.co/spaces/Jaswanth10/RAG_CHATBOT)
GitHub Repository: [GitHub](https://github.com/jaswanth123-2/RAG_CHATBOT)

Key Features

14 ML Research Papers indexed and searchable
Semantic Search with 384-dimensional embeddings
Fast LLM Responses (under 2 seconds) via Groq API
Source Citations for every answer
Interactive Chat UI built with Streamlit

System Architecture
User Query
    |
    v
Embed Query (sentence-transformers/all-MiniLM-L6-v2, 384 dims)
    |
    v
ChromaDB Vector Search (Cosine Similarity, Retrieve Top-K chunks, k=3)
    |
    v
Format Prompt (Context + Question)
    |
    v
Groq LLM API (llama-3.3-70b-versatile, Generate Answer)
    |
    v
Answer + Sources
Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | ChromaDB | Store and search embeddings |
| **Embeddings** | sentence-transformers | Convert text to vectors (384-dim) |
| **LLM** | Groq API (llama-3.3-70b) | Generate natural language answers |
| **Document Processing** | PyPDF2 | Extract text from PDF papers |
| **UI** | Streamlit | Interactive chat interface |
| **Language** | Python 3.9+ | Backend logic |

Included Research Papers

Attention Is All You Need - Transformer architecture (Vaswani et al., 2017)
BERT - Bidirectional encoder representations (Devlin et al., 2018)
GPT-3 - Language models are few-shot learners (Brown et al., 2020)
ResNet - Deep residual learning (He et al., 2015)
Adam Optimizer - Adaptive moment estimation (Kingma & Ba, 2014)
Dropout - Regularization technique (Srivastava et al., 2014)
Batch Normalization - Accelerating deep network training (Ioffe & Szegedy, 2015)
GANs - Generative adversarial networks (Goodfellow et al., 2014)
XGBoost - Scalable tree boosting (Chen & Guestrin, 2016)
Word2Vec - Efficient word embeddings (Mikolov et al., 2013)
YOLO - Real-time object detection (Redmon et al., 2015)
U-Net - Biomedical image segmentation (Ronneberger et al., 2015)
EfficientNet - Rethinking model scaling (Tan & Le, 2019)
RoBERTa - Robustly optimized BERT (Liu et al., 2019)

### Example Questions to Try:
```
"What is the attention mechanism in transformers?"
"How does BERT differ from GPT-3?"
"Explain dropout regularization and when to use it"
"What are the key innovations in ResNet?"
"How does batch normalization improve training?"
"What is the main idea behind GANs?"
```

### Sample Response:
```
Question: "What is dropout regularization?"

Answer: Dropout regularization is an extreme form of bagging 
where each model is trained on a single case and each parameter 
is strongly regularized by sharing it with other models [Source 1]. 
This encourages individual hidden units to learn useful features 
without relying on specific other units [Source 2].

Sources:
1. dropout_paper.pdf (similarity: 0.618)
2. dropout_paper.pdf (similarity: 0.567)
3. attention_is_all_you_need.pdf (similarity: 0.548)
```

## Performance Metrics

- **Embedding Dimension:** 384
- **Total Document Chunks:** Approximately 450-500
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 50 characters
- **Retrieval Time:** Under 100ms
- **LLM Response Time:** 1-2 seconds
- **Total Query Time:** Under 2.5 seconds


Acknowledgments

Research papers from ArXiv and original authors
Groq for fast LLM inference API
ChromaDB for vector database
Sentence Transformers for embeddings
Streamlit for the UI framework


Made by [Jaswanth Reddy]
