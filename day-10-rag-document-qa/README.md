# Day 10: RAG Document Q&A System

A command-line question answering system for a fictional online store, built with Retrieval-Augmented Generation (RAG). The system finds relevant information in a knowledge base and gives it to an LLM, which answers using only that information.

## Knowledge source
A fictional online store knowledge base (`knowledge_base.txt`) with 10 chunks covering shipping, returns, payment methods, order tracking, damaged items, warranty, account help, customer support hours, discounts and international shipping.

## RAG workflow
1. **Chunking:** the document is split into 10 chunks, one meaningful topic per chunk.
2. **Embedding:** each chunk is converted into a vector with the Gemini embedding model.
3. **Storage:** the vectors and their text are stored in a Qdrant vector database.
4. **Retrieval:** the user's question is embedded and the 3 most similar chunks are retrieved by cosine similarity.
5. **Augmentation:** the retrieved chunks are added to the prompt as context along with the question.
6. **Generation:** Gemini writes the answer, instructed to use only the provided context and to admit when the answer is missing.

User question → query embedding → Qdrant similarity search → retrieved chunks → LLM with context → answer

## Test questions and answers
| Question | Top retrieved chunk (score) | Generated answer |
|----------|-----------------------------|------------------|
| How long will it take to receive my order? | Shipping (0.695) | Standard delivery takes 3 to 5 days within Pakistan. |
| Can I send back something I bought if I change my mind? | Damaged items (0.713), Returns (0.710) | Yes, items can be returned within 7 days of delivery if they are unused and in the original packaging. |
| Do you accept mobile wallets? | Payment methods (0.644) | I don't have that information in my knowledge base. |
| My phone stopped working after I dropped it. Is it covered? | Warranty (0.681) | No, accidental drops are not covered. |

## Question not in the knowledge base
**Question:** Do you have a physical store in Karachi?
**Answer:** I don't have that information in my knowledge base.
**How it was handled:** The knowledge base has no information about physical stores. Retrieval still returned its 3 closest chunks (International shipping at 0.711, Payment methods and Shipping), because similarity search always returns the nearest results even when none are relevant. The system prompt tells the model to answer only from the provided context and to say it does not have the information when the answer is missing, so the model refused instead of inventing an answer.

## Observations
- Retrieval matched questions to the right chunk despite different wording. "Mobile wallets" found the Payment methods chunk (Easypaisa, JazzCash), and "dropped it" found the Warranty chunk about accidental drops.
- For the returns question, the Damaged items chunk scored 0.713 and the Returns chunk 0.710. The ranking was nearly a tie, but the LLM still picked the right information from the retrieved context.
- For "Do you accept mobile wallets?", retrieval was correct but the LLM answered "I don't have that information" because the chunk never uses the words "mobile wallet". The strict instruction to answer only from the context made the model too cautious. <If you changed the prompt: "After adjusting the prompt to allow common-sense connections while still forbidding new facts, the system answered correctly and still refused the Karachi question.">
- Similarity search always returns results, even for questions the knowledge base cannot answer (the Karachi question had a top score of 0.711, higher than some correct matches). This is why the prompt instruction matters, and why real systems also add a minimum score threshold.
- A plain LLM could not know details such as the 7-day return window or the 200 rupee delivery fee, because that information exists only in this knowledge base. RAG supplies it at question time.

## Run
From the repo root:
1. `uv sync`
2. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`
3. `uv run day-10-rag-document-qa/rag.py`