from typing import Annotated, Literal, NotRequired, Any
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


RouteName = Literal["assistant_general", "dessia_api", "ask_clarification"]


class AssistantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: NotRequired[RouteName]
    route_reason: NotRequired[str]
    dessia_tool_name: NotRequired[str]
    dessia_arguments: NotRequired[dict[str, Any]]
    missing_inputs: NotRequired[list[str]]
