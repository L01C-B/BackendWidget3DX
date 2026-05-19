import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

load_dotenv()


# def build_llm() -> ChatMistralAI:
#     model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")

#     return ChatMistralAI(
#         model=model,
#         temperature=0.1,
#         max_retries=2,
#     )

def build_llm() -> ChatOpenAI:
    model = os.getenv("GE_MODEL", "mistral.mistral-large-2402-v1:0")
    api_key = os.getenv("GE_API_KEY")
    base_url = os.getenv("GE_BASE_URL", "https://openai.generative.engine.capgemini.com/v1")

    if not api_key:
        raise RuntimeError(
            "GE_API_KEY est absente. Vérifier fichier .env."
        )
    
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.1,
        max_retries=2,
        max_tokens=1500,
    )