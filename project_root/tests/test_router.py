# tests/test_router.py
from app.request_response_router import RequestResponseRouter

queries = [
    "hi",
    "today weather is good",
    "What are SBI ATM charges?",
    "thanks"
]

for q in queries:
    print("\nUSER:", q)
    res = RequestResponseRouter.handle(q, user_acl=["ROLE_USER"])
    print("TYPE:", res["type"])
    print("RESPONSE:", res["response"])
