import logfire
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.state import AgentState
from app.config import settings


def create_checkpointer() -> BaseCheckpointSaver:
    """
    In development, use in-memory MemorySaver.
    (Postgres checkpointer commented out until Postgres database is configured).
    """
    # ============================================================
    # POSTGRES CHECKPOINTER (Commented out for development)
    # ============================================================
    # postgres_uri = getattr(settings, "postgres_uri", None) or getattr(settings, "POSTGRES_URI", None)
    # if postgres_uri:
    #     try:
    #         from langgraph.checkpoint.postgres import PostgresSaver
    #         from psycopg_pool import ConnectionPool
    #         pool = ConnectionPool(conninfo=postgres_uri, max_size=20, open=True)
    #         with PostgresSaver.from_conn_string(postgres_uri) as setup_saver:
    #             setup_saver.setup()
    #         return PostgresSaver(pool)
    #     except Exception as e:
    #         logfire.warning(f"⚠️ Postgres checkpointer unavailable ({e}); falling back to MemorySaver.")

    logfire.info("🧠 Using in-memory MemorySaver for agent checkpointing.")
    return MemorySaver()


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> StateGraph:
    """
    Build and compile the LangGraph RAG agent.
    """
    if checkpointer is None:
        checkpointer = create_checkpointer()

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)

    def route_planner(state: AgentState):
        if state["current_query"] == "CONVERSATIONAL":
            return "responder"
        return "retriever"

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {"retriever": "retriever", "responder": "responder"},
    )

    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile(checkpointer=checkpointer)
