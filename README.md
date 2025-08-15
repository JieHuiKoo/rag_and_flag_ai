# 🏗️ Country RAG System

A Retrieval-Augmented Generation (RAG) system for country information queries, built using concepts learned in `day03-rag-v2-filled.ipynb`.

## 🎯 Overview

This system allows users to ask natural language questions about countries and receive AI-generated responses based on retrieved country information. For example:

- **User**: "Which countries are hot?"
- **System**: Returns relevant hot countries like "Thailand, Singapore, Malaysia"

## 🏛️ Architecture

The system implements the RAG pattern learned in the course:

1. **Vector Database**: ChromaDB with country content embeddings
2. **Embedding Model**: intfloat/e5-base-v2
3. **LLM**: Google Flan-T5-base for response generation
4. **Data Source**: `country_content.csv` with 238 countries

## 🧠 Key Concepts Applied

Based on `day03-rag-v2-filled.ipynb`, this system demonstrates:

### 1. **Embeddings & Vector Search**
```python
embed_model_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/e5-base-v2"
)

# Query for semantic similarity (like in notebook)
results = collection.query(
    query_texts=[ question ],
    n_results=top_k
)
```

### 2. **ChromaDB Collection Management**
```python
# Create a persistent Chroma client so embedding does not need to be rerun
chroma_client = chromadb.PersistentClient(".chroma_db")

# Create collection with specified embedding function
# If the collection already exists, simply return it instead of replacing it
wiki_collection = chroma_client.get_or_create_collection(
    name="wikipedia", embedding_function=embed_model_fn
)

# Add documents to collection
for i, chunk in enumerate(all_chunks):
    wiki_collection.add(
        ids=[f"{chunk['country']}_{i}"],  # unique ID for each chunk
        documents=[chunk["content"]],  # the chunk text
        metadatas=[{"country": chunk["country"]}],  # metadata
    )
```

### 3. **Flan-T5 Integration**
```python
# Same model and generation pattern as notebook
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Using GenerationConfig for sampling
config = GenerationConfig(
    do_sample=True,
    temperature=0.7,
    top_k=50
)
```

### 4. **RAG Pipeline**
```python
# Following the notebook's prompt template pattern
prompt = f"Answer based on context:\n\n{context}\n\n{question}"

# Same encoding/generation flow
enc_prompt = tokenizer(prompt, return_tensors="pt").input_ids
enc_answer = model.generate(enc_prompt, generation_config=config)
answer = tokenizer.decode(enc_answer[0], skip_special_tokens=True)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- UV package manager (already set up)

### Installation
```bash
# Dependencies are already in pyproject.toml
uv sync
```

### Usage

#### Option 1: Interactive CLI
```bash
uv run python main.py
```

Then enter queries like:
- "Which countries are hot?"
- "Countries with mountains"
- "Island nations in Asia"
- "Countries with oil resources"

#### Option 2: Quick Demo
```bash
uv run python test_rag.py
```

#### Option 3: Automated Demo
```bash
uv run python demo.py
```

## 📊 System Features

### 🔍 **Semantic Search**
- Uses vector embeddings to find semantically similar countries
- Retrieves top-k most relevant countries based on query

### 🤖 **AI Response Generation**
- Generates natural language responses using Flan-T5
- Provides context-aware answers about countries

### 📺 **Streaming Output**
- Character-by-character streaming for better CLI experience
- Shows retrieved countries and AI reasoning

### 🗄️ **Persistent Vector Store**
- ChromaDB stores embeddings for fast retrieval
- Metadata includes country names for easy reference
****
## 📁 Project Structure

```
rag_and_flag_ai/
├── main.py                   # Main RAG system (CLI interface)
├── test_rag.py               # Quick demo with sample data
├── demo.py                   # Automated interactive demo
├── pyproject.toml            # Dependencies
├── day03-rag-v2-filled.ipynb # Course notebook with concepts
└── rag_and_flag_ai/
    ├── country_content.csv   # Country data (238 countries)
    └── other files...
```

## 🔧 Technical Details

### Dependencies
- `chromadb>=1.0.16` - Vector database
- `sentence-transformers>=5.1.0` - Embedding model
- `transformers>=4.46.0` - Flan-T5 model
- `torch>=2.0.0` - PyTorch backend
- `pandas>=2.3.1` - Data manipulation

### Performance
- **Initialization**: ~30-60 seconds (first run, creating embeddings)
- **Query Response**: ~2-5 seconds per query
- **Memory Usage**: ~2-3GB (models + embeddings)

## 🎓 Learning Outcomes

This project demonstrates mastery of:

1. **Vector Databases**: ChromaDB setup, collections, querying
2. **Embeddings**: Sentence transformers, semantic similarity
3. **Language Models**: Flan-T5, tokenization, generation
4. **RAG Architecture**: Retrieval + generation pipeline
5. **CLI Development**: Interactive command-line interfaces

## 🌟 Example Interactions

```
🗺️  Ask about countries (or 'quit' to exit): Which countries are hot?

🔍 Searching for countries related to: 'Which countries are hot?'
⏳ Retrieving relevant information...

📋 Retrieved countries: Thailand, Singapore, Malaysia, Indonesia, Philippines

🤖 AI Response:
--------------------------------------------------
Countries with hot climates include Thailand, Singapore, Malaysia, Indonesia, and Philippines. These nations are located in tropical regions with consistently warm temperatures year-round.
--------------------------------------------------
```

## 📝 Notes

- First run takes longer due to embedding generation
- Subsequent runs are faster (embeddings cached)
- System works best with country-related queries
- Responses may vary due to sampling in generation

## 🤝 Contributing

This project is based on course material from `day03-rag-v2-filled.ipynb`. Feel free to extend with:
- Different embedding models
- Alternative LLMs
- Additional data sources
- UI improvements

---

**Built with ❤️ using concepts from day03-rag-v2-filled.ipynb**