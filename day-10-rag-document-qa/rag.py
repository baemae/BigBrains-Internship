import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY. Add it to your .env file.")

client = genai.Client(api_key=API_KEY)
EMBED_MODEL = "gemini-embedding-001"   # check the docs for current names
LLM_MODEL = "gemini-3.5-flash-lite"
COLLECTION = "store_kb"
TOP_K = 3

KB_PATH = Path(__file__).parent / "knowledge_base.txt"

SYSTEM_PROMPT = (
    "You are a customer support assistant for an online store. "
    "Answer the question using ONLY the context provided. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't have that information in my knowledge base.\" "
    "Do not guess or invent details. Keep the answer short and clear."
)

TEST_QUESTIONS = [
    "How long will it take to receive my order?",
    "Can I send back something I bought if I change my mind?",
    "Do you accept mobile wallets?",
    "My phone stopped working after I dropped it. Is it covered?",
    "Do you have a physical store in Karachi?",   # NOT in the knowledge base
]


def embed(texts, task_type):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def load_chunks():
    """Chunking: split the document into one chunk per blank-line-separated paragraph."""
    text = KB_PATH.read_text(encoding="utf-8")
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def build_vector_db(chunks):
    """Embed the chunks and store them in Qdrant (in memory, since the KB is tiny)."""
    db = QdrantClient(":memory:")
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    db.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    db.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
            for i in range(len(chunks))
        ],
    )
    return db


def retrieve(db, question):
    """Retrieval: embed the question and find the most similar chunks."""
    q_vector = embed([question], "RETRIEVAL_QUERY")[0]
    hits = db.query_points(
        collection_name=COLLECTION, query=q_vector, limit=TOP_K, with_payload=True
    ).points
    return [(h.score, h.payload["text"]) for h in hits]


def generate_answer(question, retrieved):
    """Augmentation + generation: give the retrieved chunks to the LLM with the question."""
    context = "\n\n".join(text for _, text in retrieved)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )
    return response.text.strip()


def answer_question(db, question):
    retrieved = retrieve(db, question)
    print("=" * 70)
    print(f"Question: {question}")
    print("Retrieved chunks:")
    for score, text in retrieved:
        print(f"  ({score:.3f}) {text[:90]}...")
    print(f"\nAnswer: {generate_answer(question, retrieved)}\n")


def main():
    chunks = load_chunks()
    db = build_vector_db(chunks)
    print(f"Loaded {len(chunks)} chunks into the vector database.\n")

    for q in TEST_QUESTIONS:
        answer_question(db, q)

    print("Now ask your own questions (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            answer_question(db, q)


if __name__ == "__main__":
    main()