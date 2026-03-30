from pydantic import BaseModel
from typing import List, Optional

from deepeval.test_case import LLMTestCase
from agent_system.metrics_evaluator.deep_eval_llm import MyLLMWrapper
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.metrics import BaseMetric

MODEL = MyLLMWrapper(model_name="gemma3:12b")


class MetricOutputItem(BaseModel):
    metric_name: str
    score: float
    reason: Optional[str] = None


class MetricsEvaluationInputItem(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: List[str]


metrics: List[tuple[str, BaseMetric]] = [
    ("Faithfulness", FaithfulnessMetric(
        threshold=0.5,
        model=MODEL,
        include_reason=True,
        async_mode=False
    )),
    ("ResponseRelevancy", AnswerRelevancyMetric(
        threshold=0.7,
        model=MODEL,
        include_reason=True,
        async_mode=False
    )),
]


def execute_evaluate_metrics(input: MetricsEvaluationInputItem) -> List[MetricOutputItem]:
    return evaluate_samples(metrics_input=input)


def evaluate_samples(metrics_input: MetricsEvaluationInputItem) -> List[MetricOutputItem]:
    tc = LLMTestCase(
        input=metrics_input.user_input,
        retrieval_context=metrics_input.retrieved_contexts,
        actual_output=metrics_input.response
    )

    consolidated_metric_output: List[MetricOutputItem] = []

    for (name, metric) in metrics:
        metric.measure(tc)
        result_item = MetricOutputItem(
            metric_name=name,
            score=float(metric.score),
            reason=metric.reason
        )
        consolidated_metric_output.append(result_item)

    return consolidated_metric_output


def evaluate_feedback_on_metrics(input: MetricsEvaluationInputItem):
    metrics_result: List[MetricOutputItem] = execute_evaluate_metrics(input=input)

    summary = ""
    metrics_output = []

    total_score = 0.0
    for item in metrics_result:
        total_score += item.score

        if item.reason:
            summary += f"\n{item.metric_name} reason: {item.reason}"
        else:
            summary += f"\n{item.metric_name}"

        performance = "low"
        if item.score >= 0.8:
            performance = "high"
        elif item.score >= 0.6:
            performance = "medium"

        metric_item = {
            "metric_name": item.metric_name,
            "score": item.score,
            "performance": performance
        }
        metrics_output.append(metric_item)

    average_score = total_score / len(metrics_result) if metrics_result else 0.0

    overall_performance = "high"
    if average_score < 0.6:
        overall_performance = "low"
    elif average_score < 0.8:
        overall_performance = "medium"

    final_response = {
        "summary": summary.strip(),
        "metrics": metrics_output,
        "overall_score": average_score,
        "overall_performance": overall_performance,
    }

    return final_response