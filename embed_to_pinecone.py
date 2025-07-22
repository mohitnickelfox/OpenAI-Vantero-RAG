import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv
import time
import sys

load_dotenv()

# ---- Configs ----
EXCEL_PATH = "ascend.xlsx"
INDEX_NAME = "restaurant-menu-excel"
DIM = 1536  # OpenAI embedding dimension
OPENAI_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 100  # Batch size for embedding generation
UPSERT_BATCH_SIZE = 100     # Batch size for Pinecone upserts
MAX_RETRIES = 3             # Maximum retry attempts for API calls

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment variables.")
        sys.exit(1)

def load_and_validate_data():
    """Load and validate Excel data"""
    try:
        if not os.path.exists(EXCEL_PATH):
            print(f"❌ Excel file not found: {EXCEL_PATH}")
            sys.exit(1)
        
        print(f"📊 Loading data from {EXCEL_PATH}...")
        menu_df = pd.read_excel(EXCEL_PATH)
        
        if menu_df.empty:
            print("❌ Excel file is empty")
            sys.exit(1)
        
        # Check for required columns
        required_columns = ['Name']
        missing_columns = [col for col in required_columns if col not in menu_df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {', '.join(missing_columns)}")
            sys.exit(1)
        
        print(f"✅ Successfully loaded {len(menu_df)} menu items")
        print(f"📋 Available columns: {', '.join(menu_df.columns.tolist())}")
        return menu_df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        sys.exit(1)

def setup_embedder():
    """Setup OpenAI embeddings with error handling"""
    try:
        embedder = OpenAIEmbeddings(
            model=OPENAI_MODEL,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            # dimensions=512,  # Uncomment to use smaller dimensions for cost/speed
        )
        print(f"✅ OpenAI embedder initialized with model: {OPENAI_MODEL}")
        return embedder
    except Exception as e:
        print(f"❌ Error initializing OpenAI embedder: {e}")
        sys.exit(1)

def setup_pinecone_index():
    """Setup Pinecone index with error handling"""
    try:
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        cloud = "aws"
        region = "us-east-1"
        
        # Check if index exists
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_NAME not in existing_indexes:
            print(f"🔨 Creating new Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIM,
                metric='cosine',
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(10)
        else:
            print(f"✅ Using existing Pinecone index: {INDEX_NAME}")
        
        index = pc.Index(INDEX_NAME)
        
        # Check existing data
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"📊 Index currently contains {stats.total_vector_count} vectors")
            response = input("Continue and add more vectors? This will add to existing data. (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                sys.exit(0)
        
        return index
        
    except Exception as e:
        print(f"❌ Error setting up Pinecone: {e}")
        sys.exit(1)

def getval(row, col):
    """Helper function to safely get column value"""
    return str(row[col]) if col in row and pd.notnull(row[col]) else ""

def generate_documents(menu_df):
    """Generate document texts and IDs from DataFrame"""
    print("📝 Generating document texts...")
    docs, ids = [], []
    
    for i, row in tqdm(menu_df.iterrows(), total=len(menu_df), desc="Processing menu items"):
        doc_text = f"""
Food Category: {getval(row, 'Food Category')}
Meal Category: {getval(row, 'Meal Category')}
Name: {getval(row, 'Name')}
Menu Description: {getval(row, 'Menu Description')}
Description: {getval(row, 'Description')}
Allergens: {getval(row, 'Allergens')}
Chef's Description: {getval(row, "Chef's Description")}
Unique Facts: {getval(row, 'Unique Facts')}
Glossary: {getval(row, 'Glossary')}
Ingredients: {getval(row, 'Ingredients')}
Serving Details: {getval(row, 'Serving Details')}
Preparation: {getval(row, 'Preparation')}
Modifications: {getval(row, 'Modifications')}
Drop Highlights: {getval(row, 'Drop Highlights')}
""".strip()
        
        docs.append(doc_text)
        ids.append(f"menuitem-{i}")
    
    print(f"✅ Generated {len(docs)} documents")
    return docs, ids

def generate_embeddings_with_retry(embedder, docs):
    """Generate embeddings with batching and retry logic"""
    print(f"🔄 Generating embeddings in batches of {EMBEDDING_BATCH_SIZE}...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(docs), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = docs[i:i + EMBEDDING_BATCH_SIZE]
        
        for attempt in range(MAX_RETRIES):
            try:
                batch_embeddings = embedder.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"❌ Failed to generate embeddings after {MAX_RETRIES} attempts: {e}")
                    sys.exit(1)
                else:
                    print(f"⚠️ Attempt {attempt + 1} failed, retrying... Error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"✅ Generated {len(all_embeddings)} embeddings")
    return all_embeddings

def prepare_vectors(menu_df, ids, docs, embeddings):
    """Prepare vectors with metadata for Pinecone"""
    print("📦 Preparing vectors with metadata...")
    vectors = []
    
    for i in tqdm(range(len(ids)), desc="Preparing vectors"):
        row = menu_df.iloc[i]
        meta = {
            "Name": getval(row, 'Name'),
            "Food Category": getval(row, 'Food Category'),
            "Meal Category": getval(row, 'Meal Category'),
            "Menu Description": getval(row, 'Menu Description'),
            "Description": getval(row, 'Description'),
            "Allergens": getval(row, 'Allergens'),
            "Chef's Description": getval(row, "Chef's Description"),
            "Unique Facts": getval(row, 'Unique Facts'),
            "Glossary": getval(row, 'Glossary'),
            "Ingredients": getval(row, 'Ingredients'),
            "Serving Details": getval(row, 'Serving Details'),
            "Preparation": getval(row, 'Preparation'),
            "Modifications": getval(row, 'Modifications'),
            "Drop Highlights": getval(row, 'Drop Highlights'),
            "Lifelike Based Scenarios": getval(row, 'Lifelike Based Scenarios'),
            "text": docs[i]
        }
        vectors.append((ids[i], embeddings[i], meta))
    
    print(f"✅ Prepared {len(vectors)} vectors")
    return vectors

def upsert_vectors_with_retry(index, vectors):
    """Upsert vectors to Pinecone with retry logic"""
    print(f"⬆️ Upserting vectors in batches of {UPSERT_BATCH_SIZE}...")
    
    total_batches = (len(vectors) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    
    for start in tqdm(range(0, len(vectors), UPSERT_BATCH_SIZE), desc="Upserting batches"):
        batch = vectors[start:start + UPSERT_BATCH_SIZE]
        batch_num = start // UPSERT_BATCH_SIZE + 1
        
        for attempt in range(MAX_RETRIES):
            try:
                index.upsert(vectors=batch)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"❌ Failed to upsert batch {batch_num} after {MAX_RETRIES} attempts: {e}")
                    sys.exit(1)
                else:
                    print(f"⚠️ Batch {batch_num} attempt {attempt + 1} failed, retrying... Error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
    
    print(f"✅ Successfully upserted {len(vectors)} vectors")

def main():
    """Main execution function"""
    print("🚀 Starting restaurant menu embedding process...")
    print("=" * 60)
    
    # Step 1: Validate environment
    validate_environment()
    
    # Step 2: Load and validate data
    menu_df = load_and_validate_data()
    
    # Step 3: Setup embedder
    embedder = setup_embedder()
    
    # Step 4: Setup Pinecone
    index = setup_pinecone_index()
    
    # Step 5: Generate documents
    docs, ids = generate_documents(menu_df)
    
    # Step 6: Generate embeddings
    embeddings = generate_embeddings_with_retry(embedder, docs)
    
    # Step 7: Prepare vectors
    vectors = prepare_vectors(menu_df, ids, docs, embeddings)
    
    # Step 8: Upsert to Pinecone
    upsert_vectors_with_retry(index, vectors)
    
    # Step 9: Final verification
    print("\n🔍 Verifying upload...")
    final_stats = index.describe_index_stats()
    print(f"📊 Final index statistics:")
    print(f"   Total vectors: {final_stats.total_vector_count}")
    print(f"   Dimension: {final_stats.dimension}")
    
    print("\n" + "=" * 60)
    print("✅ Embedding and upsert process completed successfully!")
    print(f"🎯 Processed {len(menu_df)} menu items")
    print(f"📚 Index: {INDEX_NAME}")
    print(f"🤖 Model: {OPENAI_MODEL}")

if __name__ == "__main__":
    main()