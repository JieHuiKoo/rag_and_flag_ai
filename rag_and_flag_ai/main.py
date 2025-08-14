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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import sys
import time
from pathlib import Path


def show_example_queries():
    """Display example queries to help users"""
    examples = [
        "Which countries are hot?",
        "Countries with mountains",
        "Island nations in Asia", 
        "Countries with oil resources",
        "Desert countries",
        "European countries with coastlines",
        "Countries known for tourism",
        "Landlocked countries in Africa"
    ]
    
    print("\n💡 Example queries you can try:")
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")


class CountryRAGSystem:
    """Enhanced RAG system for country information retrieval and generation"""
    
    def __init__(self, csv_path="country_content.csv"):
        self.csv_path = csv_path
        self.embed_model_name = "BAAI/bge-small-en-v1.5"
        self.llm_model_name = "google/flan-t5-large"
        self.collection_name = "countries"
        
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
        print("📊 Loading embedding model (BAAI/bge-small-en-v1.5)...")
        try:
            self.embed_model_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embed_model_name
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise
    
    def _setup_llm(self):
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def _setup_vector_db(self):
        """Setup ChromaDB and load country data - following day03 notebook pattern"""
        print("🗄️  Setting up ChromaDB...")
        
        try:
            # Create Chroma client
            self.chroma_client = chromadb.Client()
            
            # Delete collection if it exists (like in the notebook)
            try:
                self.chroma_client.delete_collection(self.collection_name)
                print("   🔄 Cleared existing collection")
            except:
                pass
            
            # Create collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_model_fn
            )
            
            # Load country data only if collection is empty
            if self.collection.count() <= 0:
                print("📖 Loading and processing country data...")
                self._load_country_data()
            
            count = self.collection.count()
            print(f"✅ ChromaDB ready with {count} countries")
            
        except Exception as e:
            print(f"❌ Failed to setup ChromaDB: {e}")
            raise
    
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
    
    def generate_answer(self, question, top_k=5):
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
            
            for i, doc_id in enumerate(results['ids'][0]):
                chunk = self.collection.get(doc_id)
                content = chunk['documents'][0]
                country_name = chunk['metadatas'][0]['country']
                retrieved_countries.append(country_name)
                
                # Store full content for comprehensive summaries
                country_full_info[country_name] = content
                
                # Use more content for context (600 chars for better understanding)
                truncated_content = content[:600] + "..." if len(content) > 600 else content
                context += f"Country: {country_name}\n{truncated_content}\n\n"
            
            # Create enhanced prompt template for comprehensive responses
            prompt = f"""Answer the question using the provided country information.

            Context:
            {context}

            Question: {question}

            Answer format:
            ANSWER: [Your answer]
            TOP COUNTRIES: [List 3 countries]
            COUNTRY SUMMARIES:
            {retrieved_countries[0] if retrieved_countries else 'Singapore'}: [Description]
            {retrieved_countries[1] if len(retrieved_countries) > 1 else 'Indonesia'}: [Description] 
            {retrieved_countries[2] if len(retrieved_countries) > 2 else 'Malaysia'}: [Description]"""
            
            # Generate answer using Flan-T5
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,  # Increased for much longer prompts
                truncation=True,
                padding=True  # Add padding to fix attention mask warning
            )
            
            # Use generation config for comprehensive responses
            config = GenerationConfig(
                do_sample=True,
                temperature=0.3,  # Lower temperature for more consistent formatting
                top_k=20,  # Reduced for more focused responses
                max_new_tokens=350,  # Slightly reduced for better quality
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Prevent repetition
            )
            
            enc_answer = self.model.generate(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                generation_config=config
            )
            answer = self.tokenizer.decode(enc_answer[0], skip_special_tokens=True)
            
            # Return both the enhanced answer and retrieved countries with full info
            return answer, retrieved_countries, country_full_info
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Sorry, I encountered an error: {e}", [], {}
    
    def stream_generate_answer(self, question, top_k=5):
        """Generate answer with enhanced streaming output for CLI interaction"""
        print(f"\n🔍 Searching for countries related to: '{question}'")
        
        # Show thinking process
        thinking_steps = ["⏳ Retrieving relevant information", "🧠 Processing context", "🤖 Generating response"]
        for step in thinking_steps:
            print(f"   {step}...")
            time.sleep(0.5)
        
        # Get comprehensive answer and retrieved countries
        answer, countries, full_info = self.generate_answer(question, top_k)
        
        if countries:
            print(f"\n📋 Retrieved countries: {', '.join(countries)}")
        
        print(f"\n🤖 Comprehensive AI Response:")
        print("=" * 60)
        
        # Parse and display the structured response
        if "ANSWER:" in answer and "TOP COUNTRIES:" in answer and "COUNTRY SUMMARIES:" in answer:
            # Split the response into sections
            sections = answer.split("TOP COUNTRIES:")
            answer_part = sections[0].replace("ANSWER:", "").strip()
            
            remaining = sections[1] if len(sections) > 1 else ""
            if "COUNTRY SUMMARIES:" in remaining:
                countries_part = remaining.split("COUNTRY SUMMARIES:")[0].strip()
                summaries_part = remaining.split("COUNTRY SUMMARIES:")[1].strip()
            else:
                countries_part = remaining.strip()
                summaries_part = ""
            
            # Display formatted response
            print("🎯 ANSWER:")
            print("-" * 30)
            for char in answer_part:
                print(char, end='', flush=True)
                time.sleep(0.015)
            
            print(f"\n\n🌍 TOP COUNTRIES:")
            print("-" * 30)
            for char in countries_part:
                print(char, end='', flush=True)
                time.sleep(0.01)
            
            if summaries_part:
                print(f"\n\n📖 COUNTRY SUMMARIES:")
                print("-" * 30)
                for char in summaries_part:
                    print(char, end='', flush=True)
                    time.sleep(0.01)
        else:
            # Fallback to regular streaming if format not recognized
            for char in answer:
                print(char, end='', flush=True)
                time.sleep(0.02)
        
        print("\n" + "=" * 60)
        return answer, countries


def main():
    """Enhanced main CLI interface with better error handling"""
    print("=" * 60)
    print("🏗️  COUNTRY RAG SYSTEM")
    print("Based on concepts from day03-rag-v2-filled.ipynb")
    print("=" * 60)
    
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
            question = input("🗺️  Ask about countries: ").strip()
            
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
