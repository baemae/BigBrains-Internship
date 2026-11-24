import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads GEMINI_API_KEY from .env

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GEMINI_API_KEY. Add it to your .env file.")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.5-flash-lite"   # check the docs for the current model name
TEMPERATURE = 0.7            # 0 = focused/predictable, higher = more creative
MAX_OUTPUT_TOKENS = 1024   # upper limit on response length


def ask_llm(prompt: str) -> str:
    """Send the prompt to the LLM API and return the text response."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="You are a helpful assistant. Keep answers clear and concise.",
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text


def main():
    print("=== CLI AI Assistant === (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            answer = ask_llm(user_input)
            print(f"\nAI: {answer}\n")
        except Exception as e:
            print(f"\n[Error] Request failed: {e}\n")


if __name__ == "__main__":
    main()