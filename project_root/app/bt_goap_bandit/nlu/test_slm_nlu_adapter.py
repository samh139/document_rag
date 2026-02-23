from bt_goap_bandit.nlu.slm_nlu_adapter import SLMNLUAdapter
from datetime import datetime

adapter=SLMNLUAdapter()


while True:
    user_input=input("You:")
    start_time = datetime.now()
    # Format in "dd Month year hour minutes seconds"
    start_formatted_time = start_time.strftime("%d %B %Y %H:%M:%S")
    print(f"****execution start time {start_formatted_time}")
    print("sending text for analysis ",user_input)
    result=adapter.parse(text=user_input)
    print("result=",result)
    end_time=datetime.now()
    end_formatted_time = end_time.strftime("%d %B %Y %H:%M:%S")
    print(f"****execution  end time {end_formatted_time}")

