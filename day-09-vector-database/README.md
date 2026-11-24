# Day 9: Semantic Search with a Vector Database (Qdrant)

A small semantic search system. Fictional university FAQs are converted into embeddings with the Gemini embedding model (`gemini-embedding-001`), stored in a local Qdrant vector database, and searched with natural-language questions.

## Database selected: Qdrant (local mode)

Qdrant is an open-source vector database. I used its local mode, which stores the data in a folder on my machine, so it needs no account, server or payment. A normal database searches for exact values such as "price < 5000", while Qdrant finds records whose vectors are most similar in meaning to a query vector. It stores each vector together with a payload (metadata), searches by cosine similarity, and can filter results by metadata.

## How it works

Text → Gemini embedding model → vector → Qdrant collection → similarity search → relevant results

Each record stored in Qdrant contains:

- the embedding vector (3072 numbers)
- the original text
- metadata: `doc_id`, `category`, `title` and `source`

## Sample information stored

| ID | Category | Title | Text |
|----|----------|-------|------|
| faq1 | admissions | Application Deadlines | Applications for the fall semester are accepted from January to May through the online portal. |
| faq2 | fees | Tuition Fee | The tuition fee is 150,000 rupees per semester, payable in two installments. |
| faq3 | fees | Scholarships | Merit and need-based scholarships cover up to 100 percent of the tuition fee. |
| faq4 | campus | Library Hours | The library is open from 8 AM to 10 PM and students can borrow up to five books for two weeks. |
| faq5 | campus | Hostel Facilities | Hostel rooms are available for both male and female students with Wi-Fi and a mess facility. |
| faq6 | academics | Attendance Policy | Students must maintain at least 75 percent attendance to appear in final exams. |
| faq7 | academics | CS Department | The Computer Science department offers courses in programming, databases and artificial intelligence. |
| faq8 | careers | Career Services | The career services office organizes job fairs and helps students find internships. |
| faq9 | support | Password Reset | To reset your student portal password, click Forgot Password and follow the email link. |
| faq10 | campus | Transport | The campus bus service runs on six routes across the city every weekday morning. |

## Search queries and results

The questions deliberately use different words from the stored text. Scores are cosine similarity, so higher means more similar.

| Question | Top result (score) | 2nd result (score) | Relevant? |
|----------|--------------------|--------------------|-----------|
| How can I pay less for my degree? | Scholarships (0.680) | Tuition Fee (0.657) | Yes |
| Where can I stay if I live in another city? | Hostel Facilities (0.609) | Application Deadlines (0.599) | Yes |
| I can't log in to my account, what should I do? | Password Reset (0.695) | Application Deadlines (0.569) | Yes |
| Is there help finding a job after graduation? | Career Services (0.729) | Application Deadlines (0.603) | Yes |
| What happens if I miss too many classes? | Attendance Policy (0.720) | Library Hours (0.608) | Yes |

**Metadata filter test:** searching "How can I pay less for my degree?" with `category = fees` returned only Scholarships and Tuition Fee.

## Observations

- All five questions returned the correct FAQ as the top result even though they share almost no words with the stored text. For example, "stay in another city" found Hostel Facilities and "miss too many classes" found Attendance Policy.
- The 2nd and 3rd results were often weak. The "Application Deadlines" FAQ appeared in the top 3 for every question, even unrelated ones like login problems, with scores (about 0.57 to 0.64) not far below some genuinely relevant results.
- The hostel question had the smallest margin: 0.609 for the correct result against 0.599 for the next one. A small gap between scores means the ranking is less trustworthy.
- A high similarity score does not guarantee a correct answer. A real application needs extra steps such as a minimum score threshold, metadata filters (which worked well here), re-ranking and better chunking before passing results to an LLM.

## Run

From the repo root:

1. `uv sync`
2. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`
3. `uv run day-09-vector-database/vector_search.py`

The script creates a local `qdrant_db/` folder, which is git-ignored.