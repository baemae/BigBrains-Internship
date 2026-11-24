# Day 8: Embeddings and Semantic Search

A small experiment that converts text into embeddings with the Gemini embedding model (`gemini-embedding-001`) and compares semantic search with a simple keyword search.

## What is an embedding?

An embedding turns a piece of text into a list of numbers called a vector. The numbers capture the meaning of the text, so sentences that mean similar things end up with vectors that are close together, even when they use different words. In this experiment every sentence became a vector of 3072 numbers.

## Keyword search vs semantic search

- **Keyword search** looks for the same words. The question "How do I get my money back?" shares no words with "Students can request a full refund within 14 days of purchasing a course", so a keyword search cannot connect them.
- **Semantic search** compares meaning. It connects "money back" with "refund", because their embeddings are close together.

## The experiment

**Topic:** a fictional online course platform.

**Method:**
1. Wrote 10 short sentences (the knowledge base) and converted each into an embedding.
2. Wrote 5 test questions that use different words from the sentences and converted each into an embedding.
3. Compared each question with every sentence using cosine similarity (a higher score means more similar).
4. Also ran a basic keyword search (counting shared words) for comparison.

### Information used

1. Our platform offers online programming courses in Python, JavaScript and Java.
2. Students can request a full refund within 14 days of purchasing a course.
3. Every course includes a certificate of completion after passing the final quiz.
4. Live mentoring sessions are held every Saturday evening on Zoom.
5. The monthly subscription costs 20 dollars and gives access to all courses.
6. Forgot your password? Use the reset link on the login page to create a new one.
7. Our data science track covers statistics, machine learning and data visualization.
8. You can download course videos and watch them offline in our mobile app.
9. Scholarships are available for students who cannot afford the full price.
10. Support is available by email and live chat from 9 AM to 6 PM on weekdays.

### Test questions and results

| Question | Semantic search top match (score) | Keyword search best match | Better method |
|----------|-----------------------------------|---------------------------|---------------|
| Where can I learn software development online? | Online programming courses (0.698) | Online programming courses (1 shared word) | Tie |
| How do I get my money back? | Refund within 14 days (0.672) | Programming courses (0 shared words, no real match) | Semantic |
| Can I study without internet? | Download videos, watch offline (0.700) | Refund sentence (1 shared word, wrong) | Semantic |
| What if I cannot pay for the courses? | Scholarships (0.752) | Scholarships (3 shared words) | Tie |
| How can I log in if I lost my credentials? | Password reset link (0.750) | Download videos (2 shared words, wrong) | Semantic |

## Observations

- Semantic search returned the correct sentence as its top result for all 5 questions, including the ones that share almost no words with the answer. "Money back" found the refund policy, "study without internet" found offline downloads, and "lost my credentials" found the password reset.
- Keyword search failed on 3 of the 5 questions. For "money back" it found zero shared words and simply returned the first sentence. For the other two it matched on common words like "can" and "in", which have nothing to do with the question.
- Keyword search only worked when the question happened to reuse the same words as the answer, such as "online" in the first question and "cannot" in the fourth.
- The 2nd and 3rd semantic results were often weak. For "Can I study without internet?", the scholarships sentence scored 0.657, close to the correct answer's 0.700. This shows that a similarity score is a ranking and not proof of relevance, so a real system needs a score threshold and other checks before trusting a result.

## Why embeddings matter for RAG

In a RAG application, the documents are converted into embeddings once and stored. When a user asks a question, the question is also converted into an embedding and the system retrieves the documents whose embeddings are most similar. Those documents are then given to an LLM, which uses them to write an answer based on real information instead of guessing. Without embeddings the system could only find text that happens to contain the same words as the question.

## Why embeddings are useful for large amounts of information

When there are thousands of documents, a person cannot read them all and an LLM cannot take all of them as input every time. Embeddings let a computer search the whole collection by meaning and quickly pick out the few most relevant pieces. Users also do not need to guess the exact keywords used in the documents, because a question worded differently still finds the right information. Only those few relevant pieces are then passed to the LLM, which keeps answers accurate and cheaper to generate.

## Run

From the repo root:

1. `uv sync`
2. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`
3. `uv run day-08-embeddings-semantic-search/semantic_search.py`