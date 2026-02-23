from bt_goap_bandit.runtime.cli import generate_intent_agent_response

while True:
    user_text=input("You:")
    result=generate_intent_agent_response(user_input=user_text)
    print(result)