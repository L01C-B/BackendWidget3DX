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


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=request.message)
            ]
        }
    )

    last_message = result["messages"][-1]

    return ChatResponse(
        answer=last_message.content
    )