import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config ---
INDEX_NAME = "restaurant-menu-excel"

try:
    # --- Setup Pinecone Client ---
    print(f"Connecting to Pinecone index '{INDEX_NAME}'...")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    # --- Get and Print Index Stats ---
    print("\nFetching index statistics...")
    stats = index.describe_index_stats()
    
    print("\n--- Pinecone Index Report ---")
    print(f"Total Vectors in Index: {stats.get('total_vector_count', 0)}")
    
    namespaces = stats.get('namespaces')
    if not namespaces:
        print("No namespaces found. The index appears to be empty.")
    else:
        print("\nVector count by Namespace:")
        for namespace, details in namespaces.items():
            # The default namespace is represented by an empty string ''
            namespace_display = f"'{namespace}'" if namespace else "'default'"
            print(f"  - Namespace: {namespace_display} -> {details['vector_count']} vectors")
    print("---------------------------\n")

except Exception as e:
    print(f"\nAn error occurred: {e}")