import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY. Add it to your .env file.")

client = genai.Client(api_key=API_KEY)
EMBED_MODEL = "gemini-embedding-001"  # check the docs for the current name
COLLECTION = "university_faqs"

# Fictional university FAQs: doc_id, text, metadata
RECORDS = [
    ("faq1", "Applications for the fall semester are accepted from January to May through the online portal.",
     {"category": "admissions", "title": "Application Deadlines", "source": "Admissions Office"}),
    ("faq2", "The tuition fee is 150,000 rupees per semester, payable in two installments.",
     {"category": "fees", "title": "Tuition Fee", "source": "Finance Office"}),
    ("faq3", "Merit and need-based scholarships cover up to 100 percent of the tuition fee.",
     {"category": "fees", "title": "Scholarships", "source": "Finance Office"}),
    ("faq4", "The library is open from 8 AM to 10 PM and students can borrow up to five books for two weeks.",
     {"category": "campus", "title": "Library Hours", "source": "Library"}),
    ("faq5", "Hostel rooms are available for both male and female students with Wi-Fi and a mess facility.",
     {"category": "campus", "title": "Hostel Facilities", "source": "Student Affairs"}),
    ("faq6", "Students must maintain at least 75 percent attendance to appear in final exams.",
     {"category": "academics", "title": "Attendance Policy", "source": "Examination Office"}),
    ("faq7", "The Computer Science department offers courses in programming, databases and artificial intelligence.",
     {"category": "academics", "title": "CS Department", "source": "CS Department"}),
    ("faq8", "The career services office organizes job fairs and helps students find internships.",
     {"category": "careers", "title": "Career Services", "source": "Career Office"}),
    ("faq9", "To reset your student portal password, click Forgot Password and follow the email link.",
     {"category": "support", "title": "Password Reset", "source": "IT Helpdesk"}),
    ("faq10", "The campus bus service runs on six routes across the city every weekday morning.",
     {"category": "campus", "title": "Transport", "source": "Transport Office"}),
]

QUERIES = [
    "How can I pay less for my degree?",
    "Where can I stay if I live in another city?",
    "I can't log in to my account, what should I do?",
    "Is there help finding a job after graduation?",
    "What happens if I miss too many classes?",
]


def embed(texts, task_type):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def main():
    # 1. Create a local Qdrant database stored in a folder
    db = QdrantClient(path="qdrant_db")

    # 2. Embed the records
    texts = [r[1] for r in RECORDS]
    vectors = embed(texts, "RETRIEVAL_DOCUMENT")

    # 3. Create the collection (start fresh each run)
    if db.collection_exists(COLLECTION):
        db.delete_collection(COLLECTION)
    db.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=len(vectors[0]), distance=models.Distance.COSINE
        ),
    )

    # 4. Store each record: vector + payload (text and metadata)
    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i],
            payload={"doc_id": rec[0], "text": rec[1], **rec[2]},
        )
        for i, rec in enumerate(RECORDS)
    ]
    db.upsert(collection_name=COLLECTION, points=points)
    print(f"Stored {db.count(COLLECTION).count} records in the vector database.\n")

    # 5. Run similarity searches
    for query in QUERIES:
        hits = db.query_points(
            collection_name=COLLECTION,
            query=embed([query], "RETRIEVAL_QUERY")[0],
            limit=3,
            with_payload=True,
        ).points

        print("=" * 70)
        print(f"Question: {query}")
        for rank, hit in enumerate(hits, 1):
            p = hit.payload
            print(f"  {rank}. score={hit.score:.3f} | {p['category']} | {p['title']}")
            print(f"     {p['text']}")
        print()

    # 6. Bonus: metadata filter (only search the "fees" category)
    print("=" * 70)
    print("Filtered search (category = fees): How can I pay less for my degree?")
    filtered = db.query_points(
        collection_name=COLLECTION,
        query=embed(["How can I pay less for my degree?"], "RETRIEVAL_QUERY")[0],
        query_filter=models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value="fees"))]
        ),
        limit=2,
        with_payload=True,
    ).points
    for hit in filtered:
        print(f"  - {hit.payload['title']}: {hit.payload['text']}")

    db.close()


if __name__ == "__main__":
    main()