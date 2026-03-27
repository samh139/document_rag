from pydantic import BaseModel
from typing import List,Optional
import json
from agent_system.agentic.model_classes.chunk_data import ChunkData
from metrics_evaluator_input_item import MetricsEvaluationInputItem
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from agent_system.metrics_evaluator.deep_eval_llm_config import MyLLMWrapper
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric)
from deepeval.metrics import BaseMetric

MODEL = MyLLMWrapper(model_name="gemma3:12b")


metrics:List[tuple[str,BaseMetric]] = [
    ("Faithfulness", FaithfulnessMetric(threshold=0.5, model=MODEL, include_reason=True, async_mode=False)),
    ("ResponseRelevancy", AnswerRelevancyMetric(threshold=0.7, model=MODEL, include_reason=True, async_mode=False)),
   
        ]

class MetricOutputItem(BaseModel):
    metric_name:str
    score:float
    reason:Optional[str]    

# Pydantic model for input
class MetricsEvaluationInputItem(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: List[str]

def execute_evaluate_metrics(input:MetricsEvaluationInputItem):
    result=evaluate_samples(metrics_input=input)
    return result


def evaluate_samples(metrics_input: MetricsEvaluationInputItem)->List[MetricOutputItem]:
    tc=LLMTestCase(input=metrics_input.user_input,
                   retrieval_context=metrics_input.retrieved_contexts,
                   actual_output=metrics_input.response)
    consolidated_metric_output:List[MetricOutputItem]=[]
    for (name,metric) in metrics:
        metric.measure(tc)
        result_item=MetricOutputItem(metric_name=name,score=metric.score,reason=metric.reason)
        consolidated_metric_output.append(result_item)
    return consolidated_metric_output


"""
{{
    "user_query": "What is the recommended browser to open Oracle HCM?",
    "result": "It is recommend to use Chrome or Firefox to open Oracle HCM.",
    "metrics":[
        {"metric_name": "LLMContextPrecisionWithoutReference", "score": 0.85, "performance": "high"},
        {"metric_name": "ResponseRelevancy", "score": 0.9, "performance": "high"},
    ],
    "summary": "The system shows strong performance across all evaluated metrics, indicating reliable context usage, relevant responses, and high faithfulness to source material.",
    "overall_performance": "Strong"
}}

"""
def evaluate_feedback_on_metrics(input:MetricsEvaluationInputItem):
    metrics_result:List[MetricOutputItem] = execute_evaluate_metrics(input=input)
    #result["user_query"]=input.user_input
    #return agent_metrics_feedback(metrics_results=result)
    summary=""

    final_response={}
    metrics=[]
    
    total_score=0.0
    for item in metrics_result:
        if item.reason:
            summary+="\n"+item.metric_name +"reason:"+item.reason
        else:
            summary+="\n"+item.metric_name 
        performance="low"
        if item.score >.8:
            total_score+=item.score
            performance="high"
        metric_item={"metric_name":item.metric_name,"score":item.score,"performance":performance}
        metrics.append(metric_item)


    average_score=total_score/len(metrics_result)    
    overall_performance="high"
    if average_score<.6:
        overall_performance="low"
    if average_score>=.6 and average_score<=.7:
        overall_performance="medium"

    final_response["summary"]=summary   
    final_response["metrics"]=metrics
    final_response["overall_performance"] =overall_performance
    return final_response    



