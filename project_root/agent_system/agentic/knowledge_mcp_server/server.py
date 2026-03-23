from __future__ import annotations

from fastmcp import FastMCP

from agent_system.agentic.knowledge_mcp_server.tools import (
    detect_ambiguity,
    generate_clarification,
    retrieve_documents,
    synthesize_answer,
    retrieve_cluster,
)

mcp = FastMCP("KnowledgeServer")


retrieve_cluster.register(mcp)
retrieve_documents.register(mcp)
detect_ambiguity.register(mcp)
generate_clarification.register(mcp)
synthesize_answer.register(mcp)


if __name__ == "__main__":
    mcp.run()
