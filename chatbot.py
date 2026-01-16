"""
Hybrid RAG Chatbot for Jain Philosophy
Features:
1. Neo4j Graph + Vector Search for Book Knowledge
2. Fallback to LLM Internal Knowledge (Llama 3.3) if needed
3. Uses llama-3.3-70b-versatile model
"""

import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from groq import Groq

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")

# Term mappings
TERM_MAPPINGS = {
    "anekantavada": ["Anekāntavāda", "Anekānta", "non-absolutism"],
    "syadvada": ["Syādvāda", "Syād", "conditional predication"],
    "saptabhangi": ["Saptabhaṅgī", "seven-fold predication"],
    "naya": ["Nayas", "viewpoints", "7 Naya"],
    "gunasthana": ["Guṇasthānaka", "stages of spiritual development"],
    "tirthankara": ["Tīrthaṅkara", "Jina", "Arihanta"],
    "mahavira": ["Mahāvīra", "Vardhamana"],
    "jiva": ["Jīvāstikāya", "soul"],
    "ajiva": ["Ajīva", "non-soul"],
    "karma": ["Karma", "karmic matter"],
}

def search_neo4j_comprehensive(driver, embedder, query: str) -> List[Dict]:
    """
    Enhanced Neo4j Search Strategy:
    1. Concept Search (Fuzzy & Exact)
    2. Vector Search (Chunks)
    3. Keyword/Text Search (Fulltext)
    4. Chapter/Section Title Search
    """
    expanded_terms = []
    # Simple query expansion
    query_lower = query.lower()
    for term, variants in TERM_MAPPINGS.items():
        if term in query_lower:
            expanded_terms.extend(variants)
    
    embedding = embedder.encode(query).tolist()
    chunks = []
    
    with driver.session() as session:
        # 1. Concept Node Search (High Priority)
        try:
            result = session.run("""
                CALL db.index.fulltext.queryNodes('concept_name_index', $q)
                YIELD node, score
                RETURN 
                    'Concept: ' + node.name + ' (' + coalesce(node.category, 'General') + ')\n' + 
                    'Variants: ' + coalesce(toString(node.variants), 'None') as text, 
                    score + 1.0 as score
                LIMIT 3
            """, q=query)
            chunks.extend([dict(r) for r in result])
        except Exception: 
            pass 

        # 2. Gunasthana Specific Search
        try:
            result = session.run("""
                MATCH (g:Gunasthana)
                WHERE toLower(g.sanskrit_name) CONTAINS toLower($q) 
                   OR toLower(g.english_name) CONTAINS toLower($q)
                RETURN g.sanskrit_name + ' (' + g.english_name + ')\n' + g.description as text, 2.0 as score
            """, q=query)
            chunks.extend([dict(r) for r in result])
        except: pass

        # 3. Vector Search
        indexes = ['chunk_embeddings', 'gunasthana_embeddings']
        for idx in indexes:
            try:
                result = session.run(f"""
                    CALL db.index.vector.queryNodes('{idx}', 7, $emb)
                    YIELD node, score
                    RETURN coalesce(node.text, node.description) as text, score
                """, emb=embedding)
                chunks.extend([dict(r) for r in result])
            except: continue

        # 4. Fulltext Keyword Search
        lucene_query = query.replace("?", "").replace("!", "")
        if lucene_query.strip():
            try:
                result = session.run("""
                    CALL db.index.fulltext.queryNodes('chunk_text_index', $q)
                    YIELD node, score
                    RETURN node.text as text, score LIMIT 5
                """, q=lucene_query)
                chunks.extend([dict(r) for r in result])
            except: pass

        # 5. Structure/Chapter Search (Table of Contents)
        # If the user asks for "chapters", "summary", "outline", "structure"
        structure_keywords = ["chapter", "summary", "outline", "structure", "table of contents", "book"]
        if any(k in query_lower for k in structure_keywords):
            try:
                # Fetch all chapters sorted by number
                result = session.run("""
                    MATCH (c:Chapter)
                    RETURN c.number as number, c.title as title
                    ORDER BY c.number ASC
                """)
                chapters = [f"Chapter {r['number']}: {r['title']}" for r in result]
                if chapters:
                    toc_text = "Book Table of Contents (All Chapters):\n" + "\n".join(chapters)
                    chunks.append({
                        "text": toc_text,
                        "score": 2.5 # Very high relevance for structural questions
                    })
            except: pass

    # Deduplicate and Sort
    seen = set()
    unique_chunks = []
    
    # Sort by score descending
    for c in sorted(chunks, key=lambda x: x['score'], reverse=True):
        content = c['text']
        # Simple dedupe (using first 100 chars signature)
        sig = content[:100] if content else ""
        if sig and sig not in seen:
            seen.add(sig)
            unique_chunks.append(c)
    
    # Return top results. 
    # If we have the TOC (score 2.5), it will be at the top.
    return unique_chunks[:7]

class HybridRetriever:
    def __init__(self, driver, embedder):
        self.driver = driver
        self.embedder = embedder

    def search_book(self, query: str) -> List[Dict]:
        return search_neo4j_comprehensive(self.driver, self.embedder, query)

def ask_jain_sage(user_query: str, retriever: HybridRetriever, client: Groq) -> str:
    """
    Call llama-3.3-70b-versatile directly with book context + internal knowledge fallback.
    """
    # 1. Retrieve from Book
    book_chunks = retriever.search_book(user_query)
    book_text = "\n\n".join([c['text'] for c in book_chunks])
    
    system_prompt = (
        "You are an expert scholar on Jain philosophy. "
        "Use the provided context from the book 'Anekant Syadvad' to answer the question. "
        "If the book context is insufficient, use your own broad knowledge of Jainism and religion to answer comprehensively. "
        "Do NOT mention 'According to the text' just give the answer naturally. "
        "Always define Sanskrit terms."
        "Ensure the response is logically structured, concise yet comprehensive, and suitable for both "
        "academic and general readers."
        "If the available book context is partial or insufficient, responsibly supplement the answer "
        "using well-established principles of Jain philosophy and comparative religious knowledge, "
        "without introducing speculation. "
        "Whenever Sanskrit or Prakrit terms appear, always: "
        "1) Write the term in standard IAST-style transliteration, "
        "2) Clearly define the term in simple and precise language at its first occurrence. "
        "Use the following transliteration standard consistently: "
        "Vowels: "
        "अ a, आ ā, इ i, ई ī, उ u, ऊ ū, ऋ ṛ, ए e, ऐ ai, ओ o, औ au, अं ṁ/ṅ, अः ḥ. "
        "Consonants: "
        "क् k, ख् kh, ग् g, घ् gh, ङ् ṅ; "
        "च् c, छ् ch, ज् j, झ् jh, ञ् ñ; "
        "ट् ṭ, ठ् ṭh, ड् ḍ, ढ् ḍh, ण् ṇ; "
        "त् t, थ् th, द् d, ध् dh, न् n; "
        "प् p, फ् ph, ब् b, भ् bh, म् m; "
        "य् y, र् r, ल् l, व् v; "
        "श् ś, ष् ṣ, स् s, ह् h. "
    )

    user_message_content = f"Context from Book:\n{book_text}\n\nQuestion: {user_query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message_content}
    ]

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_completion_tokens=2048,
        top_p=0.95,
        

    )

    return completion.choices[0].message.content

def main():
    print("="*60)
    print("  Jain Philosophy AI Expert")
    print("  (Neo4j Graph + Llama 3.3 Internal Knowledge)")
    print("="*60)

    try:
        # Init Resources
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        embedder = SentenceTransformer('all-mpnet-base-v2')
        retriever = HybridRetriever(driver, embedder)
        
        # Init Groq Client
        client = Groq(api_key=GROQ_API_KEY)
        
        print("\n✓ System Ready")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return

    # Loop
    while True:
        try:
            q = input("\nQ: ").strip()
            if q.lower() in ['exit', 'quit']: break
            if not q: continue
            
            print("  Thinking...", end='\r')
            ans = ask_jain_sage(q, retriever, client)
            print(" "*30, end='\r')
            print(f"A: {ans}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

    driver.close()

if __name__ == "__main__":
    main()
