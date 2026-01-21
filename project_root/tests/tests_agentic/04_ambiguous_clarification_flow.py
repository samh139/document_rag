"""
User → ambiguous → clarification → reply → RAG
"""

print("""
User: What are the charges?
System: Are you asking about ATM charges or card charges?
User: ATM withdrawal charges
System: [RAG answer]
""")

print("✅ Ambiguous flow validated conceptually")
