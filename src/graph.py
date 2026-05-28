import logging
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from src.llm import build_llm
from src.prompts import SYSTEM_PROMPT, DESSIA_FORMAT_PROMPT, ROUTER_PROMPT, DESSIA_KEYWORDS
from src.state import AssistantState
from src.tools.dessia_client import run_dessia_analysis

logger = logging.getLogger("copilot3dx")

llm = build_llm()


# récupération message utilisateur
def get_last_user_message(state: AssistantState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            content = message.content

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return " ".join(str(item) for item in content)

            return str(content)

    return ""


# choix route par le LLM
def router_node(state: AssistantState) -> dict:
    messages = [
        SystemMessage(content=ROUTER_PROMPT),
        *state["messages"][-5:],
    ]

    response = llm.invoke(messages)

    logger.info("[router] raw_llm_response=%s", response.content)

    try:
        decision = json.loads(response.content.replace("\\_", "_"))
    except Exception:
        logger.exception("[router] JSON parsing error")    
        return {
            "route": "ask_clarification",
            "route_reason": "Le routeur LLM n'a pas retourné un JSON valide.",
            "missing_inputs": [
                "La requête n'a pas pu être interprétée correctement. Merci de reformuler ou de répéter les paramètres du workflow."
            ]
        }


    logger.info("[router] decision=%s", json.dumps(decision, ensure_ascii=False))

    return {
        "route": decision.get("route", "assistant_general"),
        "route_reason": decision.get("reason", ""),
        "dessia_tool_name": decision.get("tool_name"),
        "dessia_arguments": decision.get("arguments", {}),
        "missing_inputs": decision.get("missing_inputs", []),
    }



# choix de la prochaine étape
def choose_next_node(state: AssistantState) -> str:
    return state.get("route", "assistant_general")


# l'assistant général
def assistant_general_node(state: AssistantState) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# en cas de manque d'informations
def ask_clarification_node(state: AssistantState) -> dict:
    missing_inputs = state.get("missing_inputs", [])

    content = (
        "Il me manque certaines informations pour lancer l'analyse Dessia :\n"
        + "\n".join(f"- {item}" for item in missing_inputs)
    )

    return {
        "messages": [AIMessage(content=content)]
    }


# outil : dessia
def dessia_api_node(state: AssistantState) -> dict:
    user_message = get_last_user_message(state)

    tool_name = state.get("dessia_tool_name")
    arguments = state.get("dessia_arguments", {})

    dessia_result = run_dessia_analysis(tool_name, arguments)

    logger.info("[dessia] raw_dessia_result=%s", json.dumps(dessia_result, ensure_ascii=False))

    messages = [
        SystemMessage(content=DESSIA_FORMAT_PROMPT),
        HumanMessage(
            content=(
                "Voici la demande utilisateur :\n"
                f"{user_message}\n\n"
                "Voici l'outil Dessia utilisé :\n"
                f"{tool_name}\n\n"
                "Voici le résultat retourné par le service Dessia :\n"
                f"{dessia_result}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


###################################################################################

# assemblage du graphe
def build_graph():
    builder = StateGraph(AssistantState)

    ###

    builder.add_node("router", router_node)
    builder.add_node("assistant_general", assistant_general_node)   
    builder.add_node("ask_clarification", ask_clarification_node)
    builder.add_node("dessia_api", dessia_api_node)

    ###

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        choose_next_node,
        {
            "assistant_general": "assistant_general",
            "dessia_api": "dessia_api",
            "ask_clarification": "ask_clarification", 
        },
    )

    builder.add_edge("assistant_general", END)
    builder.add_edge("ask_clarification", END)
    builder.add_edge("dessia_api", END)

    return builder.compile()


graph = build_graph()