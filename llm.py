from groq import Groq
from config import GROQ_API_KEY


client = Groq(
    api_key=GROQ_API_KEY
)


def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Use only the context below to answer.

Context:
{context}

Question:
{question}

Answer:
"""


    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )


    return response.choices[0].message.content