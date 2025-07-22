import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---- Configs ----
EXCEL_PATH = "ascend.xlsx"
INDEX_NAME = "restaurant-menu-excel"
DIM = 1536  # OpenAI `text-embedding-3-large` model output dimension
OPENAI_MODEL = "text-embedding-3-small"  # You can use text-embedding-3-small for smaller vectors

# ---- Load Menu Items ----
menu_df = pd.read_excel(EXCEL_PATH)

# ---- Setup OpenAI Embeddings ----
embedder = OpenAIEmbeddings(
    model=OPENAI_MODEL,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# ---- Setup Pinecone v3 Client ----
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
cloud = "aws"
region = "us-east-1"

# ---- Create Pinecone Index If Not Exists ----
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = pc.Index(INDEX_NAME)

# ---- Generate Document Texts ----
docs, ids = [], []
for i, row in tqdm(menu_df.iterrows(), total=len(menu_df)):
    def getval(col):
        return str(row[col]) if col in row and pd.notnull(row[col]) else ""
    
    doc_text = f"""
Food Category: {getval('Food Category')}
Meal Category: {getval('Meal Category')}
Name: {getval('Name')}
Menu Description: {getval('Menu Description')}
Description: {getval('Description')}
Allergens: {getval('Allergens')}
Chef's Description: {getval("Chef's Description")}
Unique Facts: {getval('Unique Facts')}
Glossary: {getval('Glossary')}
Ingredients: {getval('Ingredients')}
Serving Details: {getval('Serving Details')}
Preparation: {getval('Preparation')}
Modifications: {getval('Modifications')}
Drop Highlights: {getval('Drop Highlights')}
"""
    docs.append(doc_text.strip())
    ids.append(f"menuitem-{i}")

# ---- Get Embeddings from OpenAI ----
embeddings = embedder.embed_documents(docs)

# ---- Prepare and Upsert to Pinecone ----
vectors = []
for i in range(len(ids)):
    row = menu_df.iloc[i]
    meta = {
        "Name": str(row['Name']) if 'Name' in row else "",
        "Food Category": str(row['Food Category']) if 'Food Category' in row else "",
        "Meal Category": str(row['Meal Category']) if 'Meal Category' in row else "",
        "Menu Description": str(row['Menu Description']) if 'Menu Description' in row else "",
        "Description": str(row['Description']) if 'Description' in row else "",
        "Allergens": str(row['Allergens']) if 'Allergens' in row else "",
        "Chef's Description": str(row["Chef's Description"]) if "Chef's Description" in row else "",
        "Unique Facts": str(row['Unique Facts']) if 'Unique Facts' in row else "",
        "Glossary": str(row['Glossary']) if 'Glossary' in row else "",
        "Ingredients": str(row['Ingredients']) if 'Ingredients' in row else "",
        "Serving Details": str(row['Serving Details']) if 'Serving Details' in row else "",
        "Preparation": str(row['Preparation']) if 'Preparation' in row else "",
        "Modifications": str(row['Modifications']) if 'Modifications' in row else "",
        "Drop Highlights": str(row['Drop Highlights']) if 'Drop Highlights' in row else "",
        "Lifelike Based Scenarios": str(row['Lifelike Based Scenarios']) if 'Lifelike Based Scenarios' in row else "",
        "text": docs[i]
    }
    vectors.append((ids[i], embeddings[i], meta))

# ---- Batch Upsert ----
for start in range(0, len(vectors), 100):
    batch = vectors[start:start + 100]
    index.upsert(vectors=batch)

print("✅ Embedding and upsert complete using OpenAI.")
