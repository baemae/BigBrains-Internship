# Day 7: LLM CLI Assistant

A minimal command-line assistant that sends user input to the Gemini API and prints the response.

## Run
From the repo root:
1. `uv sync`
2. Copy `.env.example` to `.env` and add your Gemini API key
3. `uv run day-07-llm-cli-assistant/main.py`

## Security
The API key lives in `.env`, which is git-ignored and never pushed.

## Settings experiments
- **temperature:** <what you saw at 0.0 vs 1.5>
- **max_output_tokens:** <what happened at 50>

## How the request flows
User input → `ask_llm()` → Gemini SDK builds an HTTPS request (model, prompt, system instruction, settings, API key) → Gemini model generates a response → API returns it → SDK extracts the text → printed in the terminal.