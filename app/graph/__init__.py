"""LangGraph orchestration.

    state.py    the strongly-typed AgentState
    builder.py  graph assembly (nodes, edges, extension points)
    nodes/      one file per node

Every node calls exactly one service and merges the result into state. No node
issues a SQL query, touches a repository, or talks to Qdrant/OpenRouter
directly.
"""
