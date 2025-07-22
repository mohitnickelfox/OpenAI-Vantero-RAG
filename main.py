import os
import random
import json
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Changed from ChatGroq
from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()
# --- Config ---
PINECONE_INDEX = "restaurant-menu-excel"
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SCENARIOS_CSV = "scenarios.csv"
PERSONAS_CSV = "personas.csv"
ADDITIONAL_CONTEXT_CSV = "additional_context.csv"
QUESTIONS_CSV = "questions_with_feedback.csv"
OPENAI_MODEL = "text-embedding-3-small"

# --- Initializations ---
embedder = OpenAIEmbeddings(
    model=OPENAI_MODEL,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

# --- LLM Switched to Gemini ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# --- Updated Prompt with Few-Shot Examples ---
QUESTION_PROMPT = """
You are an expert restaurant trainer bot. Your goal is to generate a high-quality, challenging training question.

Based on past feedback, here are examples of what to do and what to avoid. Learn from them.
{feedback_examples}

Now, using the following NEW details, generate a completely new question that meets the highest standards.

RELEVANT SCENARIO: {scenario}
RELEVANT PERSONA: {persona}
RELEVANT ADDITIONAL CONTEXT: {additional_context}
MENU CONTEXT: {menu_context}

Instructions: Use only the new inputs to craft a question that is both strictly relevant and challenging for waitstaff. The question must require reasoning across all provided details.

Respond ONLY with a valid JSON as follows:
{{
  "question": "...",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct_answer": "A",
  "reasoning": "...",
  "menu_items_referenced": ["..."]
}}
Requirements:
- Use ONLY menu items and details from the provided context.
- The question must relate directly to the provided scenario, persona, and additional context.
- Create practical, scenario-based questions.
- Include 4 plausible options (A, B, C, D) with only ONE correct answer.
- Reference at least 2 different menu items.
- Ensure the reasoning for the correct answer is brief and explains both why the question was generated and why the answer is correct.
- Keep the question, options, and reasoning short, crisp, and to the point.
"""

# --- Helper Functions (Original and New) ---

def load_titles(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp1252')
    df.columns = df.columns.str.strip()
    if 'title' not in df.columns:
        raise Exception(f"CSV {file_path} missing 'title' column.")
    return df['title'].dropna().astype(str).tolist()

def embed(text: str) -> np.ndarray:
    return np.array(embedder.embed_query(text)).reshape(1, -1)

def find_relevant(item: str, candidates: List[str]) -> str:
    if not candidates:
        return ""
    item_vec = embed(item)
    c_vecs = np.vstack([embed(t) for t in candidates])
    sims = cosine_similarity(item_vec, c_vecs)[0]
    best_idx = np.argmax(sims)
    return candidates[best_idx]

def get_menu_context(query: str, k: int = 5) -> List[Dict]:
    query_vec = embedder.embed_query(query)
    response = index.query(vector=query_vec, top_k=k, include_metadata=True, include_values=False)
    docs = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        parts = []
        for key in [
            "Name", "Food Category", "Meal Category", "Menu Description",
            "Description", "Ingredients", "Allergens", "Unique Facts",
            "Fun Fact", "Serving Details", "Modifications", "Drop Highlights"
        ]:
            if meta.get(key):
                parts.append(f"{key.upper()}: {meta[key]}")
        docs.append({"content": "\n".join(parts), "metadata": meta})
    return docs

def extract_json_response(raw: str) -> Optional[Dict]:
    # This handles potential markdown ```json ... ``` added by the model
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
        
    try:
        return json.loads(raw)
    except Exception:
        print("Failed to extract JSON from LLM output:\n", raw)
        return None

def get_user_feedback() -> (int, str):
    """Gets a quality rating and notes from the user."""
    while True:
        try:
            rating = int(input("Rate the quality of this question (1-5, where 5 is best): ").strip())
            if 1 <= rating <= 5:
                break
            else:
                print("Invalid rating. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    notes = input("What could be improved? (e.g., 'options too easy', 'not relevant'): ").strip()
    return rating, notes

def get_feedback_examples() -> str:
    """Reads the CSV and returns formatted good and bad examples to guide the LLM."""
    if not os.path.exists(QUESTIONS_CSV):
        return "No past feedback available."
        
    df = pd.read_csv(QUESTIONS_CSV)
    if len(df) < 2:
        return "Not enough feedback yet to provide examples."

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    df_sorted = df.sort_values(by='rating', ascending=False)
    
    examples = []
    
    good_example = df_sorted.iloc[0]
    examples.append(
        "### GOOD QUESTION EXAMPLE (Rating: 5/5) ###\n"
        f"CONTEXT: Scenario: {good_example['scenario']}, Persona: {good_example['persona']}\n"
        f"GENERATED QUESTION: {good_example['question']}\n"
        f"REASONING: This is a great question because it connects the persona's needs directly to specific menu details.\n"
    )
    
    bad_example = df_sorted.iloc[-1]
    if bad_example['rating'] < 3:
        examples.append(
            "### POOR QUESTION EXAMPLE (Rating: 1/5) ###\n"
            f"CONTEXT: Scenario: {bad_example['scenario']}, Persona: {bad_example['persona']}\n"
            f"GENERATED QUESTION: {bad_example['question']}\n"
            f"REASONING FOR POOR RATING: {bad_example['feedback_notes']}\n"
            "AVOID THIS: This question was poorly rated. Do not make similar mistakes.\n"
        )
    
    return "\n".join(examples)

def save_question(question_data: Dict, combo: Dict, rating: int, feedback_notes: str):
    """Saves the generated question along with its user-provided feedback."""
    header = [
        "id", "scenario", "persona", "additional_context", "question",
        "option_a", "option_b", "option_c", "option_d", "correct_answer",
        "reasoning", "menu_items_referenced", "created_at", "rating", "feedback_notes"
    ]
    if not os.path.exists(QUESTIONS_CSV):
        with open(QUESTIONS_CSV, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
    with open(QUESTIONS_CSV, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        row_id = f"Q_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        q = question_data
        writer.writerow([
            row_id, combo["scenario"], combo["persona"], combo["additional_context"],
            q.get("question", ""), q.get("options", {}).get("A", ""),
            q.get("options", {}).get("B", ""), q.get("options", {}).get("C", ""),
            q.get("options", {}).get("D", ""), q.get("correct_answer", ""),
            q.get("reasoning", ""), ", ".join(q.get("menu_items_referenced", [])),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            rating, feedback_notes
        ])

def build_prompt(combo: Dict) -> str:
    """Builds the final prompt, including dynamically retrieved feedback examples."""
    feedback_examples = get_feedback_examples()
    return QUESTION_PROMPT.format(
        feedback_examples=feedback_examples,
        scenario=combo['scenario'],
        persona=combo['persona'],
        additional_context=combo['additional_context'],
        menu_context=combo['menu_context']
    )

def generate_strictly_relevant_combo(menu_context: str, scenarios: List[str], personas: List[str], additional_contexts: List[str]) -> Dict:
    relevant_additional = find_relevant(menu_context, additional_contexts)
    composite = menu_context + " " + relevant_additional
    relevant_scenario = find_relevant(composite, scenarios)
    persona_context = menu_context + " " + relevant_scenario
    relevant_persona = find_relevant(persona_context, personas)
    return {
        "menu_context": menu_context, "scenario": relevant_scenario,
        "persona": relevant_persona, "additional_context": relevant_additional
    }

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("\n🍽️ Restaurant Question Generator with Feedback Loop 🍽️\n")
    scenarios = load_titles(SCENARIOS_CSV)
    personas = load_titles(PERSONAS_CSV)
    additional_contexts = load_titles(ADDITIONAL_CONTEXT_CSV)

    if not all([scenarios, personas, additional_contexts]):
        print("Error: One or more context CSVs are missing or empty.")
        exit(1)

    while True:
        menu_query = random.choice([
            "starters", "main course", "dessert", "beverage", "gluten", "dairy",
            "vegan", "vegetarian", "nut free", "chef's special", "modifications"
        ])
        print(f"\nSearching for menu items related to: '{menu_query}'...")
        menu_docs = get_menu_context(menu_query)
        if not menu_docs:
            print(f"No menu docs found for query: {menu_query}. Trying another one.")
            continue
        menu_context = "\n".join([doc["content"] for doc in menu_docs])

        combo = generate_strictly_relevant_combo(menu_context, scenarios, personas, additional_contexts)
        prompt = build_prompt(combo)
        
        print(f"\nBuilding prompt with feedback... Sending to Gemini...")

        llm_response = llm.invoke(prompt)
        result_json = extract_json_response(
            llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        )
        if not result_json:
            print("LLM did not return valid JSON, skipping.")
            continue
            
        print("\nQUESTION GENERATED:\n", json.dumps(result_json, indent=2))
        
        # --- Human-in-the-Loop Feedback ---
        rating, feedback_notes = get_user_feedback()
        save_question(result_json, combo, rating, feedback_notes)
        print("✅ Question and feedback saved.\n")
        
        cont = input("Do you want to generate another question? (y/n): ").strip().lower()
        if cont != "y":
            break

    print(f"All done! Questions and feedback written to: {QUESTIONS_CSV}")




