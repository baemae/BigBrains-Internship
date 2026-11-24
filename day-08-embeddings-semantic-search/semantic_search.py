import math
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY. Add it to your .env file.")

client = genai.Client(api_key=API_KEY)
EMBED_MODEL = "gemini-embedding-001"  # check the docs for the current name

# Knowledge base: a fictional online course platform
DOCUMENTS = [
    "Our platform offers online programming courses in Python, JavaScript and Java.",
    "Students can request a full refund within 14 days of purchasing a course.",
    "Every course includes a certificate of completion after passing the final quiz.",
    "Live mentoring sessions are held every Saturday evening on Zoom.",
    "The monthly subscription costs 20 dollars and gives access to all courses.",
    "Forgot your password? Use the reset link on the login page to create a new one.",
    "Our data science track covers statistics, machine learning and data visualization.",
    "You can download course videos and watch them offline in our mobile app.",
    "Scholarships are available for students who cannot afford the full price.",
    "Support is available by email and live chat from 9 AM to 6 PM on weekdays.",
]

QUESTIONS = [
    "Where can I learn software development online?",
    "How do I get my money back?",
    "Can I study without internet?",
    "What if I cannot pay for the courses?",
    "How can I log in if I lost my credentials?",
]


def embed(texts, task_type):
    """Convert a list of texts into embedding vectors."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def keyword_search(question, docs):
    """Naive keyword search: count shared words, for comparison."""
    q_words = set(question.lower().replace("?", "").split())
    scores = [len(q_words & set(d.lower().replace(".", "").split())) for d in docs]
    best = max(range(len(docs)), key=lambda i: scores[i])
    return best, scores[best]


def main():
    doc_vectors = embed(DOCUMENTS, "RETRIEVAL_DOCUMENT")
    print(f"Embedded {len(DOCUMENTS)} documents. Vector length: {len(doc_vectors[0])}\n")

    for question in QUESTIONS:
        q_vector = embed([question], "RETRIEVAL_QUERY")[0]
        scores = [cosine_similarity(q_vector, d) for d in doc_vectors]
        top3 = sorted(range(len(DOCUMENTS)), key=lambda i: scores[i], reverse=True)[:3]

        print("=" * 70)
        print(f"Question: {question}")
        print("Semantic search, top 3:")
        for rank, i in enumerate(top3, 1):
            print(f"  {rank}. ({scores[i]:.3f}) {DOCUMENTS[i]}")

        k_idx, k_score = keyword_search(question, DOCUMENTS)
        print(f"Keyword search best match ({k_score} shared words):")
        print(f"     {DOCUMENTS[k_idx]}\n")


if __name__ == "__main__":
    main()