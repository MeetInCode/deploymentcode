"""
Hybrid RAG Ingestion Pipeline for "Anekant Syadvad" - Jain Philosophy Book

This script implements a sophisticated knowledge graph-based RAG system with:
1. Hierarchical chunking (Book → Chapter → Section → Chunk)
2. Rich metadata extraction
3. Entity recognition for key Jain concepts
4. Proper Unicode/transliteration handling
5. Semantic embeddings with contextual overlap
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
FILE_PATH = "Anekant Syadvad - Final.txt"

# Key Jain Philosophy Concepts for Entity Recognition
JAIN_CONCEPTS = {
    # Core Doctrines
    "Anekāntavāda": ["Anekantavada", "Anekāntvāda", "Anekānta", "many-sidedness", "non-absolutism"],
    "Syādvāda": ["Syadvada", "Syād", "conditional predication", "theory of conditioned predication"],
    "Saptabhaṅgī": ["Saptabhangi", "seven-fold predication", "seven propositions"],
    "Naya": ["Nayas", "viewpoints", "perspectives", "7 Naya", "seven viewpoints"],
    
    # Six Substances (Dravya)
    "Jīvāstikāya": ["Jivastikaya", "Jīva", "Jiva", "soul", "living being", "consciousness"],
    "Pudgalāstīkāya": ["Pudgalastikaya", "Pudgala", "matter"],
    "Dharmāstikāya": ["Dharmastikaya", "Dharma", "motion principle"],
    "Adharmāstikāya": ["Adharmastikaya", "Adharma", "rest principle"],
    "Ākāśāstikāya": ["Akashastikaya", "Ākāṡa", "Akasha", "space"],
    "Kāla": ["Kala", "time"],
    
    # Nine Tattvas
    "Tattvas": ["Tattva", "nine elements", "fundamental principles"],
    "Āsrava": ["Asrava", "influx of karma"],
    "Bandha": ["Bandh", "bondage of karma"],
    "Saṃvara": ["Samvara", "stoppage"],
    "Nirjarā": ["Nirjara", "shedding of karma"],
    "Mokṣa": ["Moksha", "liberation", "salvation"],
    "Punya": ["merit", "good karma"],
    "Pāpa": ["Papa", "demerit", "sin"],
    
    # Religious Terms
    "Tīrthaṅkara": ["Tirthankara", "Tīrthankara", "Jina", "Arhat", "Arihaṅta"],
    "Mahāvīra": ["Mahavira", "Vardhamana"],
    "Ṛṣabhadeva": ["Rishabhadeva", "Adinatha", "first Tirthankara"],
    "Karma": ["Karmas", "karmic matter"],
    "Saṅgha": ["Sangha", "fourfold community"],
    "Sādhu": ["Sadhu", "monk"],
    "Sādhvī": ["Sadhvi", "nun"],
    "Śrāvaka": ["Shravak", "layman"],
    "Śrāvikā": ["Shravika", "laywoman"],
    
    # Types of Knowledge
    "Kēvalajñāna": ["Kevala Jnana", "Kevalajnana", "omniscience", "absolute knowledge"],
    "Mati Jñāna": ["Mati Jnana", "sensory knowledge"],
    "Śruta Jñāna": ["Shruta Jnana", "scriptural knowledge"],
    "Avadhi Jñāna": ["Avadhi Jnana", "clairvoyance"],
    "Manaḥparyaya": ["Manahparyaya", "telepathy"],
    
    # Seven Nayas
    "Naigama Naya": ["Naigama", "common viewpoint"],
    "Saṅgraha Naya": ["Sangraha Naya", "collective viewpoint"],
    "Vyavahāra Naya": ["Vyavahara Naya", "practical viewpoint"],
    "Ṛjusūtra Naya": ["Rijusutra Naya", "linear viewpoint"],
    "Śabda Naya": ["Shabda Naya", "verbal viewpoint"],
    "Samabhirūḍha Naya": ["Samabhirudha Naya", "etymological viewpoint"],
    "Evambhūta Naya": ["Evambhuta Naya", "actuality viewpoint"],
    
    # Practices
    "Ahiṃsā": ["Ahimsa", "non-violence"],
    "Anuvratas": ["Anuvrata", "small vows", "five vows"],
    "Namaskāra Mahāmantra": ["Namaskar Mantra", "Navkar Mantra"],
    "Guṇasthānaka": ["Gunasthana", "stages of spiritual development"],
}

# Chapter structure from the book's table of contents
CHAPTERS = [
    {"number": 1, "title": "The Quest for Truth", "start_page": 1, "end_page": 13},
    {"number": 2, "title": "Perspectives and Paradoxes", "start_page": 14, "end_page": 19},
    {"number": 3, "title": "Beyond Judgement", "start_page": 20, "end_page": 26},
    {"number": 4, "title": "Jainism: A First Look", "start_page": 27, "end_page": 39},
    {"number": 5, "title": "Religion and Philosophy", "start_page": 40, "end_page": 48},
    {"number": 6, "title": "Anekāntavāda", "start_page": 49, "end_page": 65},
    {"number": 7, "title": "Syādvāda", "start_page": 66, "end_page": 75},
    {"number": 8, "title": "Four Bases", "start_page": 76, "end_page": 84},
    {"number": 9, "title": "Five Reasons", "start_page": 85, "end_page": 100},
    {"number": 10, "title": "Various aspects of Knowledge", "start_page": 101, "end_page": 115},
    {"number": 11, "title": "7 Naya", "start_page": 116, "end_page": 139},
    {"number": 12, "title": "A point of view", "start_page": 140, "end_page": 147},
    {"number": 13, "title": "Saptabhaṅgī", "start_page": 148, "end_page": 167},
    {"number": 14, "title": "Barrister Chakravarti", "start_page": 168, "end_page": 181},
    {"number": 15, "title": "Five Types of Knowledge", "start_page": 182, "end_page": 198},
    {"number": 16, "title": "Karma", "start_page": 199, "end_page": 222},
    {"number": 17, "title": "Development of the Soul", "start_page": 223, "end_page": 256},
    {"number": 18, "title": "Life is a hassle", "start_page": 257, "end_page": 282},
    {"number": 19, "title": "Confirmation and Refutation", "start_page": 283, "end_page": 292},
    {"number": 20, "title": "Namaskāra Mahāmantra", "start_page": 293, "end_page": 309},
]


class ChunkMetadata(BaseModel):
    """Pydantic model for chunk metadata"""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    chapter_number: int = Field(description="Chapter number (1-20)")
    chapter_title: str = Field(description="Title of the chapter")
    page_number: str = Field(description="Page number or range")
    chunk_index: int = Field(description="Index of chunk within chapter")
    total_chunks_in_chapter: int = Field(default=0, description="Total chunks in this chapter")
    word_count: int = Field(description="Number of words in chunk")
    has_sanskrit_terms: bool = Field(default=False, description="Whether chunk contains Sanskrit/transliterated terms")
    key_concepts: List[str] = Field(default_factory=list, description="Key Jain concepts mentioned")
    

@dataclass
class ProcessedChunk:
    """Represents a processed text chunk with metadata"""
    text: str
    metadata: ChunkMetadata
    embedding: List[float] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Clean text while preserving Unicode/transliterated characters"""
    # Remove form feed and other control characters
    text = text.replace('\x0c', '')
    text = text.replace('\x00', '')
    
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def extract_page_number(chunk: str) -> Optional[str]:
    """Extract page number from chunk"""
    lines = chunk.strip().split('\n')
    if not lines:
        return None
    
    last_line = lines[-1].strip()
    
    # Check for numeric page number
    if last_line.isdigit() and len(last_line) <= 3:
        return last_line
    
    # Check for Roman numerals (for front matter)
    if re.match(r'^[ivxlcIVXLC]+$', last_line) and len(last_line) <= 5:
        return last_line.lower()
    
    return None


def identify_chapter(page_num: str) -> Optional[Dict]:
    """Identify which chapter a page belongs to"""
    try:
        page_int = int(page_num)
        for chapter in CHAPTERS:
            if chapter["start_page"] <= page_int <= chapter["end_page"]:
                return chapter
    except (ValueError, TypeError):
        pass
    return None


def extract_key_concepts(text: str) -> List[str]:
    """Extract key Jain concepts from text"""
    found_concepts = []
    text_lower = text.lower()
    
    for main_term, variants in JAIN_CONCEPTS.items():
        # Check main term (case-insensitive for ASCII, exact for Unicode)
        if main_term.lower() in text_lower or main_term in text:
            found_concepts.append(main_term)
            continue
        
        # Check variants
        for variant in variants:
            if variant.lower() in text_lower:
                found_concepts.append(main_term)
                break
    
    return list(set(found_concepts))


def has_transliterated_terms(text: str) -> bool:
    """Check if text contains Sanskrit transliteration characters"""
    transliteration_chars = set("āīūṛṅñṭḍṇśṣḥṁĀĪŪṚṄÑṬḌṆŚṢḤṀ")
    return any(c in transliteration_chars for c in text)


def semantic_chunk(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Semantic chunking that respects paragraph and sentence boundaries.
    Uses a sliding window approach with overlap for context preservation.
    """
    # Split into paragraphs first
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds max size, save current chunk
        if len(current_chunk) + len(para) + 2 > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from end of previous
            words = current_chunk.split()
            overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
            current_chunk = " ".join(overlap_words) + "\n\n" + para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def parse_book(file_path: str) -> List[ProcessedChunk]:
    """Parse the book into hierarchical chunks with rich metadata"""
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Verify transliteration characters are present
    test_chars = ["ā", "ī", "ū", "ṛ", "ṅ", "ś", "ṣ", "ḥ"]
    found_chars = [c for c in test_chars if c in content]
    print(f"✓ Found transliterated characters: {found_chars}")
    
    # Split by form feed (page breaks)
    raw_pages = content.split('\x0c')
    print(f"Found {len(raw_pages)} raw pages")
    
    # Group content by chapter
    chapter_content: Dict[int, List[Tuple[str, str]]] = {c["number"]: [] for c in CHAPTERS}
    front_matter = []
    
    for page in raw_pages:
        page = clean_text(page)
        if not page:
            continue
        
        page_num = extract_page_number(page)
        chapter = identify_chapter(page_num) if page_num else None
        
        if chapter:
            # Remove page number from end for cleaner text
            lines = page.split('\n')
            if lines and lines[-1].strip() == page_num:
                page = '\n'.join(lines[:-1])
            
            chapter_content[chapter["number"]].append((page_num, page))
        else:
            front_matter.append((page_num or "fm", page))
    
    # Process chapters into chunks
    all_chunks: List[ProcessedChunk] = []
    
    for chapter in CHAPTERS:
        chapter_pages = chapter_content[chapter["number"]]
        if not chapter_pages:
            continue
        
        # Combine all text from chapter
        chapter_text = "\n\n".join([p[1] for p in chapter_pages])
        page_range = f"{chapter_pages[0][0]}-{chapter_pages[-1][0]}" if len(chapter_pages) > 1 else chapter_pages[0][0]
        
        # Semantic chunking
        text_chunks = semantic_chunk(chapter_text, max_chunk_size=1500, overlap=200)
        
        for idx, chunk_text in enumerate(text_chunks):
            key_concepts = extract_key_concepts(chunk_text)
            
            metadata = ChunkMetadata(
                chunk_id=f"ch{chapter['number']:02d}_chunk{idx:03d}",
                chapter_number=chapter["number"],
                chapter_title=chapter["title"],
                page_number=page_range,
                chunk_index=idx,
                total_chunks_in_chapter=len(text_chunks),
                word_count=len(chunk_text.split()),
                has_sanskrit_terms=has_transliterated_terms(chunk_text),
                key_concepts=key_concepts
            )
            
            all_chunks.append(ProcessedChunk(text=chunk_text, metadata=metadata))
    
    # Update total chunks per chapter
    for chunk in all_chunks:
        chunk.metadata.total_chunks_in_chapter = len([
            c for c in all_chunks 
            if c.metadata.chapter_number == chunk.metadata.chapter_number
        ])
    
    print(f"Created {len(all_chunks)} semantic chunks across {len(CHAPTERS)} chapters")
    return all_chunks


def create_graph_schema(session):
    """Create Neo4j indexes and constraints"""
    queries = [
        # Constraints
        "CREATE CONSTRAINT book_title IF NOT EXISTS FOR (b:Book) REQUIRE b.title IS UNIQUE",
        "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.chapter_id IS UNIQUE", 
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (con:Concept) REQUIRE con.name IS UNIQUE",
        
        # Vector index for semantic search
        """CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (n:Chunk)
        ON (n.embedding)
        OPTIONS {indexConfig: {
          `vector.dimensions`: 768,
          `vector.similarity_function`: 'cosine'
        }}""",
        
        # Full-text index for keyword search
        "CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text, n.chapter_title]",
        
        # Index for concept lookup
        "CREATE INDEX concept_name_index IF NOT EXISTS FOR (c:Concept) ON (c.name)",
    ]
    
    for query in queries:
        try:
            session.run(query)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"Warning: {e}")


def ingest_to_neo4j(chunks: List[ProcessedChunk], embedder: SentenceTransformer):
    """Ingest chunks into Neo4j with graph relationships"""
    print(f"Connecting to Neo4j at {NEO4J_URI}...")
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    driver.verify_connectivity()
    print("✓ Connected to Neo4j")
    
    with driver.session() as session:
        # Clear existing data
        print("Clearing existing book data...")
        session.run("MATCH (n) WHERE n:Book OR n:Chapter OR n:Chunk OR n:Concept DETACH DELETE n")
        
        # Create schema
        print("Creating graph schema...")
        create_graph_schema(session)
        
        # Create Book node
        session.run("""
            CREATE (b:Book {
                title: 'Anekāntavāda: The Heart of Jainism',
                author: 'Late Mr. Chandulal S. Shah',
                translator: 'Ms. Nimisha Vora',
                language: 'English',
                genre: 'Philosophy',
                total_chapters: $total_chapters
            })
        """, total_chapters=len(CHAPTERS))
        
        # Create Chapter nodes
        for chapter in CHAPTERS:
            session.run("""
                MATCH (b:Book {title: 'Anekāntavāda: The Heart of Jainism'})
                CREATE (c:Chapter {
                    chapter_id: $chapter_id,
                    number: $number,
                    title: $title,
                    start_page: $start_page,
                    end_page: $end_page
                })
                CREATE (b)-[:HAS_CHAPTER]->(c)
            """, 
                chapter_id=f"chapter_{chapter['number']}",
                number=chapter["number"],
                title=chapter["title"],
                start_page=chapter["start_page"],
                end_page=chapter["end_page"]
            )
        
        # Create sequential chapter relationships
        session.run("""
            MATCH (c1:Chapter), (c2:Chapter)
            WHERE c2.number = c1.number + 1
            CREATE (c1)-[:NEXT_CHAPTER]->(c2)
        """)
        
        # Create Concept nodes
        print("Creating concept nodes...")
        for concept_name, variants in JAIN_CONCEPTS.items():
            session.run("""
                CREATE (con:Concept {
                    name: $name,
                    variants: $variants,
                    category: $category
                })
            """, 
                name=concept_name, 
                variants=variants,
                category=categorize_concept(concept_name)
            )
        
        # Ingest chunks with embeddings
        print("Ingesting chunks with embeddings...")
        prev_chunk_id = None
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embedder.encode(chunk.text).tolist()
            chunk.embedding = embedding
            
            # Create Chunk node
            session.run("""
                MATCH (ch:Chapter {number: $chapter_number})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    text: $text,
                    chapter_number: $chapter_number,
                    chapter_title: $chapter_title,
                    page_number: $page_number,
                    chunk_index: $chunk_index,
                    word_count: $word_count,
                    has_sanskrit_terms: $has_sanskrit_terms,
                    key_concepts: $key_concepts,
                    embedding: $embedding
                })
                CREATE (ch)-[:CONTAINS]->(c)
            """,
                chunk_id=chunk.metadata.chunk_id,
                text=chunk.text,
                chapter_number=chunk.metadata.chapter_number,
                chapter_title=chunk.metadata.chapter_title,
                page_number=chunk.metadata.page_number,
                chunk_index=chunk.metadata.chunk_index,
                word_count=chunk.metadata.word_count,
                has_sanskrit_terms=chunk.metadata.has_sanskrit_terms,
                key_concepts=chunk.metadata.key_concepts,
                embedding=embedding
            )
            
            # Create NEXT relationship for sequential reading
            if prev_chunk_id:
                session.run("""
                    MATCH (c1:Chunk {chunk_id: $prev_id}), (c2:Chunk {chunk_id: $curr_id})
                    CREATE (c1)-[:NEXT]->(c2)
                """, prev_id=prev_chunk_id, curr_id=chunk.metadata.chunk_id)
            
            prev_chunk_id = chunk.metadata.chunk_id
            
            # Link to concepts
            for concept_name in chunk.metadata.key_concepts:
                session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id}), (con:Concept {name: $concept_name})
                    CREATE (c)-[:MENTIONS]->(con)
                """, chunk_id=chunk.metadata.chunk_id, concept_name=concept_name)
            
            if (i + 1) % 20 == 0:
                print(f"  Ingested {i + 1}/{len(chunks)} chunks...", end='\r')
        
        # Create concept co-occurrence relationships
        print("\nCreating concept co-occurrence relationships...")
        session.run("""
            MATCH (c1:Concept)<-[:MENTIONS]-(chunk:Chunk)-[:MENTIONS]->(c2:Concept)
            WHERE id(c1) < id(c2)
            WITH c1, c2, count(chunk) as co_occurrences
            WHERE co_occurrences > 1
            MERGE (c1)-[r:CO_OCCURS_WITH]->(c2)
            SET r.count = co_occurrences
        """)
    
    driver.close()
    print(f"\n✓ Ingestion complete! {len(chunks)} chunks stored in Neo4j.")


def categorize_concept(concept_name: str) -> str:
    """Categorize a concept for better organization"""
    categories = {
        "Core Doctrine": ["Anekāntavāda", "Syādvāda", "Saptabhaṅgī", "Naya"],
        "Six Substances": ["Jīvāstikāya", "Pudgalāstīkāya", "Dharmāstikāya", "Adharmāstikāya", "Ākāśāstikāya", "Kāla"],
        "Nine Tattvas": ["Tattvas", "Āsrava", "Bandha", "Saṃvara", "Nirjarā", "Mokṣa", "Punya", "Pāpa"],
        "Religious Terms": ["Tīrthaṅkara", "Mahāvīra", "Ṛṣabhadeva", "Karma", "Saṅgha", "Sādhu", "Sādhvī", "Śrāvaka", "Śrāvikā"],
        "Types of Knowledge": ["Kēvalajñāna", "Mati Jñāna", "Śruta Jñāna", "Avadhi Jñāna", "Manaḥparyaya"],
        "Seven Nayas": ["Naigama Naya", "Saṅgraha Naya", "Vyavahāra Naya", "Ṛjusūtra Naya", "Śabda Naya", "Samabhirūḍha Naya", "Evambhūta Naya"],
        "Practices": ["Ahiṃsā", "Anuvratas", "Namaskāra Mahāmantra", "Guṇasthānaka"],
    }
    
    for category, concepts in categories.items():
        if concept_name in concepts:
            return category
    return "Other"


def main():
    print("=" * 60)
    print("Hybrid RAG Ingestion for Anekant Syadvad")
    print("=" * 60)
    
    # Load embedding model
    print("\nLoading embedding model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    print("✓ Model loaded")
    
    # Parse book
    print("\nParsing book...")
    chunks = parse_book(FILE_PATH)
    
    # Ingest to Neo4j
    print("\nIngesting to Neo4j...")
    ingest_to_neo4j(chunks, embedder)
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
