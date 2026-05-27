import logging
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from src.graph import graph

app = FastAPI(title="3DX Copilot Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard à l'origine du widget 3DX
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Logging
# =========================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("copilot3dx")

# =========================
# Mémoire conversationnelle
# =========================

# Stockage en RAM :
# clé = thread_id
# valeur = liste de messages LangChain/ LangGraph
conversation_store: dict[str, list] = {}

# Pour éviter les accès concurrents incohérents
conversation_store_lock = Lock()

# Nombre maximum de messages conservés dans le contexte envoyé au graphe
MAX_HISTORY_MESSAGES = 20


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    thread_id: str


@app.get("/")
def root():
    return {"status": "ok", "service": "3DX Copilot Agent API"}


@app.get("/debug/memory")
def debug_memory():
    """
    Endpoint optionnel pour visualiser l'état de la mémoire.
    Utile pour debug uniquement.
    """
    with conversation_store_lock:
        return {
            "thread_count": len(conversation_store),
            "threads": {
                thread_id: len(messages)
                for thread_id, messages in conversation_store.items()
            }
        }


@app.delete("/debug/memory/{thread_id}")
def clear_thread_memory(thread_id: str):
    """
    Endpoint optionnel pour supprimer la mémoire d'un thread.
    Pratique pour tests.
    """
    with conversation_store_lock:
        existed = thread_id in conversation_store
        if existed:
            del conversation_store[thread_id]

    return {
        "thread_id": thread_id,
        "deleted": existed
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    thread_id = request.thread_id or "default-thread"
    user_message = request.message.strip()

    if not user_message:
        return ChatResponse(
            answer="Votre message est vide.",
            thread_id=thread_id
        )

    logger.info(
        "POST /chat - thread_id=%s - message=%s",
        thread_id,
        user_message
    )

    # 1) Charger l'historique existant
    with conversation_store_lock:
        history = conversation_store.get(thread_id, []).copy()

    # 2) Limiter l'historique transmis au graphe
    trimmed_history = history[-MAX_HISTORY_MESSAGES:]

    # 3) Ajouter le nouveau message utilisateur
    input_messages = trimmed_history + [HumanMessage(content=user_message)]

    logger.info(
        "Thread %s - historique chargé: %d messages (trim à %d)",
        thread_id,
        len(history),
        len(trimmed_history)
    )

    try:
        # 4) Appeler le graphe avec tout l'historique utile
        result = graph.invoke({
            "messages": input_messages
        })

        # 5) Récupérer l'état mis à jour
        updated_messages = result["messages"]
        last_message = updated_messages[-1]

        # 6) Sauvegarder l'état complet mis à jour pour ce thread
        with conversation_store_lock:
            conversation_store[thread_id] = updated_messages

        logger.info(
            "Thread %s - réponse générée - total messages stockés: %d",
            thread_id,
            len(updated_messages)
        )

        return ChatResponse(
            answer=last_message.content,
            thread_id=thread_id
        )

    except Exception as e:
        logger.exception(
            "Erreur pendant le traitement du thread %s: %s",
            thread_id,
            str(e)
        )
        return ChatResponse(
            answer="Une erreur est survenue pendant le traitement de votre demande.",
            thread_id=thread_id
        )