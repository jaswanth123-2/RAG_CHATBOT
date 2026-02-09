# 🤖 RAG Chatbot - AI-Powered Research Paper Q&A

> Ask questions about machine learning research papers using Retrieval-Augmented Generation (RAG)

## 🎯 Live Demo

- **HuggingFace Spaces:** [Demo Link]
- **GitHub Repository:** [https://github.com/jaswanth123-2/RAG_CHATBOT]

## 🌟 Features

- ✅ **14 ML Research Papers** indexed and searchable
- ✅ **Semantic Search** with 384-dimensional embeddings
- ✅ **Fast LLM Responses** (<2 seconds) via Groq API
- ✅ **Source Citations** for every answer
- ✅ **Interactive Chat UI** built with Streamlit

## 📊 System Architecture
```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Embed Query     │  sentence-transformers/all-MiniLM-L6-v2
│  (384 dims)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  ChromaDB        │  Cosine Similarity Search
│  Vector Search   │  Retrieve Top-K chunks (k=3)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Format Prompt   │  Context + Question
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Groq LLM API    │  llama-3.3-70b-versatile
│  Generate Answer │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Answer +        │
│  Sources         │
└──────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | ChromaDB | Store and search embeddings |
| **Embeddings** | sentence-transformers | Convert text to vectors (384-dim) |
| **LLM** | Groq API (llama-3.3-70b) | Generate natural language answers |
| **Document Processing** | PyPDF2 | Extract text from PDF papers |
| **UI** | Streamlit | Interactive chat interface |
| **Language** | Python 3.9+ | Backend logic |

## 📚 Included Research Papers

1. **Attention Is All You Need** - Transformer architecture (Vaswani et al., 2017)
2. **BERT** - Bidirectional encoder representations (Devlin et al., 2018)
3. **GPT-3** - Language models are few-shot learners (Brown et al., 2020)
4. **ResNet** - Deep residual learning (He et al., 2015)
5. **Adam Optimizer** - Adaptive moment estimation (Kingma & Ba, 2014)
6. **Dropout** - Regularization technique (Srivastava et al., 2014)
7. **Batch Normalization** - Accelerating deep network training (Ioffe & Szegedy, 2015)
8. **GANs** - Generative adversarial networks (Goodfellow et al., 2014)
9. **XGBoost** - Scalable tree boosting (Chen & Guestrin, 2016)
10. **Word2Vec** - Efficient word embeddings (Mikolov et al., 2013)
11. **YOLO** - Real-time object detection (Redmon et al., 2015)
12. **U-Net** - Biomedical image segmentation (Ronneberger et al., 2015)
13. **EfficientNet** - Rethinking model scaling (Tan & Le, 2019)
14. **RoBERTa** - Robustly optimized BERT (Liu et al., 2019)

## 📝 Usage Examples

### Example Questions to Try:
❓ "Explain dropout regularization and when to use it"
❓ "What are the key innovations in ResNet?"
❓ "How does batch normalization improve training?"
❓ "What is the main idea behind GANs?"
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

## 🎯 Performance Metrics

- **Embedding Dimension:** 384
- **Total Document Chunks:** ~450-500
- **Chunk Size:** 500 characters
- **Chunk Overlap:** 50 characters
- **Retrieval Time:** <100ms
- **LLM Response Time:** 1-2 seconds
- **Total Query Time:** <2.5 seconds
