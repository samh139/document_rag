# nlu/huggingface_nlu_adapter.py
import logging
from transformers import pipeline

logger = logging.getLogger("bt_goap_bandit.nlu.hf_nlu_adapter")

class HuggingFaceNLUAdapter:
    """Local HF-based NLU adapter: intent via zero-shot, entities via NER."""

    def __init__(self):
        logger.info("[HF-NLU] Loading Hugging Face pipelines …")
        self.intent_pipe = pipeline("zero-shot-classification",
                                    model="facebook/bart-large-mnli")
        self.ner_pipe = pipeline("ner", model="dslim/bert-base-NER",
                                 aggregation_strategy="simple")
        self.labels = [
            "employment_update_issue",
            "multi_query_employee_self_service",
            "promotion_request",
            "policy_query",
            "generic_query",
        ]

    def analyze(self, text: str):
        logger.info(f"[HF-NLU] → analyzing: {text}")
        intent_res = self.intent_pipe(text, self.labels)
        intent = intent_res["labels"][0]
        conf = float(intent_res["scores"][0])
        entities = [
            {"text": e["word"], "label": e["entity_group"]}
            for e in self.ner_pipe(text)
        ]
        return {"intent": intent, "confidence": conf, "entities": entities}
