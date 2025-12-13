from app.agents.engagement.engagement_agent import EngagementAgent

queries = [
    "hi",
    "thanks for the help",
    "What are SBI ATM charges?",
    "Explain what a savings account is"
]

for q in queries:
    intent = EngagementAgent.classify(q)
    print(f"{q}  -->  {intent}")

