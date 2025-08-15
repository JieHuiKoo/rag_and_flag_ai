import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import the exported CSV
# df = pd.read_csv("rag_and_flag_ai/countries_subset.csv")
df = pd.read_csv("rag_and_flag_ai/country_content.csv")
loader = CSVLoader("rag_and_flag_ai/country_content.csv")
data = loader.load()
# Initialize splitter (adjust chunk_size and overlap as needed)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

all_chunks = []
for idx, row in df.iterrows():
    country = row["Country"]
    text = row["Content"]
    # Split text into chunks
    chunks = splitter.split_text(text)
    # Optionally, add metadata for each chunk
    for chunk in chunks:
        all_chunks.append({"country": country, "content": chunk})

# Now, all_chunks is ready for embedding
# Create embedding function
embed_model_name = "intfloat/e5-base-v2"
embed_model_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    embed_model_name
)
# Chroma client
chroma_client = chromadb.PersistentClient(".chroma_db")

# Create collection with specified embedding function
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

# Check number of documents in collection
print(wiki_collection.count())
