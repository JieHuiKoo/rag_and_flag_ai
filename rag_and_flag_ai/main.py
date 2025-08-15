#!/usr/bin/env python3
"""
Optimized Country RAG System with Enhanced User Experience
Based on concepts learned in day03-rag-v2-filled.ipynb
"""

import os
import pandas as pd
import chromadb
from uuid import uuid4
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import time
from pathlib import Path


def show_example_queries():
    """Display example queries to help users"""
    examples = [
        "Country with the highest chinese population",
        "Countries with mountains",
        "Island nations in Asia", 
        "Countries with oil resources",
        "European countries with coastlines",
    ]
    
    print("\n💡 Example queries you can try:")
    for i, example in enumerate(examples, 0):
        print(f"   {i+1}. {example}")


class CountryRAGSystem:
    """Enhanced RAG system for country information retrieval and generation"""
    
    def __init__(self, csv_path="country_content.csv"):
        self.csv_path = csv_path
        self.embed_model_name = "intfloat/e5-base-v2"
        self.llm_model_name = "google/flan-t5-large"
        self.collection_name = "wikipedia"
        
        # Initialize components
        self.chroma_client = None
        self.collection = None
        self.embed_model_fn = None
        self.model = None
        self.tokenizer = None
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the complete RAG system"""
        print("🚀 Initializing Country RAG System...")
        print("   (This may take a moment on first run)")
        
        try:
            self._setup_embeddings()
            self._setup_llm()
            self._setup_vector_db()
            
            print("✅ System initialization complete!")
                
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            sys.exit(1)
    
    def _setup_embeddings(self):
        print(f"📊 Loading embedding model {self.embed_model_name}...")
        try:
            self.embed_model_fn = SentenceTransformer(self.embed_model_name
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _setup_llm(self):
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.llm_model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.llm_model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_vector_db(self):
        """Setup ChromaDB and load country data - following day03 notebook pattern"""
        print("🗄️  Setting up ChromaDB...")
        
        # try:
            # Create Chroma client
        import os
        dir = os.getcwd()
        chroma_path = dir + "/.chroma_db"
        print(f"   📂 Using ChromaDB path: {chroma_path}")
        self.chroma_client = chromadb.PersistentClient(chroma_path)
        
        # Delete collection if it exists (like in the notebook)
        # try:
        #     self.chroma_client.delete_collection(self.collection_name)
        #     print("   🔄 Cleared existing collection")
        # except:
        #     pass
        
        # Create collection
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name,
            # embedding_function=self.embed_model_fn
        )
        
        # Load country data only if collection is empty
        if self.collection.count() <= 0:
            print("📖 Loading and processing country data...")
            self._load_country_data()
        
        count = self.collection.count()
        print(f"✅ ChromaDB ready")
            
        # except Exception as e:
        #     print(f"❌ Failed to setup ChromaDB: {e}")
        #     raise
    
    def _load_country_data(self):
        """Load country data from CSV and add to ChromaDB"""
        try:
            # Check if CSV exists
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"Country data file not found: {self.csv_path}")
            
            # Read CSV file
            df = pd.read_csv(self.csv_path)
            print(f"   📄 Read {len(df)} countries from CSV")
            
            # Prepare texts and IDs (following day03 notebook pattern)
            texts = df['Content'].tolist()
            countries = df['Country'].tolist()
            
            # Generate UUIDs for IDs (like in notebook)
            texts_ids = [str(uuid4())[:8] for _ in range(len(texts))]
            
            # Create metadata with country names
            metadatas = [{"country": country} for country in countries]
            
            print(f"   ⚡ Creating embeddings for {len(texts)} countries...")
            print("      (This may take 30-60 seconds...)")
            
            # Add documents to collection (like in notebook)
            self.collection.add(
                documents=texts,
                ids=texts_ids,
                metadatas=metadatas
            )
            
            print(f"   ✅ Successfully loaded {len(texts)} countries")
            
        except Exception as e:
            print(f"❌ Failed to load country data: {e}")
            raise
    
    def generate_answer(self, question, top_k=3):
        """
        Generate answer using RAG pipeline - following day03 notebook pattern
        1. Query ChromaDB for relevant countries
        2. Combine contexts
        3. Use Flan-T5 to generate response
        """
        try:
            # Get relevant countries from ChromaDB
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k
            )
            
            # Combine contexts (following notebook pattern)
            context = ""
            retrieved_countries = []
            country_full_info = {}
            
            # Get the most relevant country only (first result)
            doc_id = results['ids'][0][0]  # Get first (most relevant) result
            chunk = self.collection.get(doc_id)
            content = chunk['documents'][0]

            for id, country_id in enumerate(results['ids'][0]):
                chunk = self.collection.get(country_id)
                country_name = chunk['metadatas'][0]['country']
                retrieved_countries.append(country_name)
            
            # Store full content for comprehensive summaries
            country_full_info[country_name] = content
            
            # Use more content for context (500 chars for better understanding)
            truncated_content = content[:500] + "..." if len(content) > 500 else content
            context = f"Country: {country_name}\n{truncated_content}\n\n"

            # Create enhanced prompt template for comprehensive responses
            prompt = f"""Generate a summary that is relevant to the question for the country using the information provided.

                Question: {question}
            
                Answer: {retrieved_countries[0]}

                Country's information:
                {context}

                Brief Summary about the country related to the question:"""
            
            # Generate answer using Flan-T5
            input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
            
            outputs = self.model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.1
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return both the enhanced answer and retrieved countries with full info
            return answer, retrieved_countries, country_full_info
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error: {e}", [], {}
    
    def stream_generate_answer(self, question, top_k=5):
        """Generate answer with enhanced streaming output for CLI interaction"""
        print(f"\n🔍 Searching for countries related to: '{question}'")
        
        # Get comprehensive answer and retrieved countries
        answer, countries, full_info = self.generate_answer(question, top_k)
        
        if countries:
            print(f"\n📋 Retrieved {len(countries)} countries for context: {', '.join(countries)}")
        
        print(f"\n🤖 LLM Response:")
        print("=" * 60)
        
        # Simply stream the raw answer from the LLM
        for char in answer:
            print(char, end='', flush=True)
            time.sleep(0.02)
        
        print("\n" + "=" * 60)
        return answer, countries


def main():
    """Enhanced main CLI interface with better error handling"""
    print("=" * 60)
    print("🏗️  COUNTRY RAG SYSTEM")
    
    try:
        # Initialize RAG system
        rag_system = CountryRAGSystem()
        
        # Show example queries
        show_example_queries()
        
        print(f"\n💬 Ready for your questions!")
        print("   Type 'examples' to see more examples")
        print("   Type 'quit' or 'exit' to quit")
        
        while True:
            print("\n" + "=" * 60)
            question = input("🗺️  Ask about countries: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the Country RAG System!")
                break
            
            if question.lower() in ['examples', 'help']:
                show_example_queries()
                continue
            
            if not question:
                print("❓ Please enter a question!")
                continue
            
            try:
                # Stream generate response
                start_time = time.time()
                rag_system.stream_generate_answer(question, top_k=5)
                end_time = time.time()
                
                print(f"⏱️  Response time: {end_time - start_time:.1f}s")
                
            except KeyboardInterrupt:
                print("\n⏸️  Query interrupted. Try again or type 'quit' to exit.")
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                print("💡 Try rephrasing your question or check if it's country-related.")
                
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("💡 Make sure all dependencies are installed: uv sync")
        sys.exit(1)


if __name__ == "__main__":
    main()
