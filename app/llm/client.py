from openai import OpenAI
from app.config import OPENAI_API_KEY, MODEL_NAME, TEMPERATURE, MAX_TOKENS

client = OpenAI(api_key=OPENAI_API_KEY)


def ask_llm(question: str) -> str:
    system_prompt = """
    You are a professional AI assistant.

    Rules:
    - Answer clearly and concisely.
    - If you are unsure, say "I don't know".
    - Do not fabricate information.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return response.choices[0].message.content
def chat(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content
