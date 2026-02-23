from bt_goap_bandit.runtime.cli import generate_goap_bandit_payload
result=generate_goap_bandit_payload(text="i need info",want_tree=True)
print("intent agent payload", result)