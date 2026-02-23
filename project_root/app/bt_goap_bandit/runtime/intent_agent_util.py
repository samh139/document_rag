from model_classes.intent_agent_response import Subplan

from app_logger import get_app_logger
from app_configs.app_env import app_env
from enum import Enum

logger=get_app_logger("intent_agent_util")



class RetrievalType(Enum):
    FULL_RETRIEVAL="FULL_RETRIEVAL"
    MINIMAL_RETRIEVAL="MINIMAL_RETRIEVAL"
    NO_RETRIEVAL="AMBIGUOUS_QUERY"

def should_retrieval_answer_be_created(subplan:Subplan)->bool:
    if subplan.bandit_arm.lower() in ["clarification_request","neutral_smalltalk"]:
        return False


    if  not subplan.bandit_arm.lower() in [
        "data_probe","analytical_reasoning",
        "task_execution","knowledge_reasoning"]:
        logger.info(f"not retrieveing since bandit_arm is {subplan.bandit_arm} ")
        return False
    
    if subplan.goap_plan.readiness_score < app_env.get_acceptable_chunks_threshold_score():
        logger.info(f"readiness score is less {subplan.goap_plan.readiness_score} not suitable for retrieval, current threshold is {app_env.get_acceptable_chunks_threshold_score()}")
        return False
    
    return True


def decide_retrieval_type(subplan:Subplan)->RetrievalType:

    if subplan.bandit_arm.lower() in ["neutral_smalltalk"]:
        return RetrievalType.NO_RETRIEVAL

    if subplan.bandit_arm.lower() in ["clarification_request"]:
        return RetrievalType.MINIMAL_RETRIEVAL

    if  not subplan.bandit_arm.lower() in [
        "data_probe","analytical_reasoning",
        "task_execution","knowledge_reasoning"]:
        logger.info(f"not retrieveing since bandit_arm is {subplan.bandit_arm} ")
        return RetrievalType.MINIMAL_RETRIEVAL
    
    if subplan.goap_plan.readiness_score < app_env.get_acceptable_chunks_threshold_score():
        logger.info(f"readiness score is less {subplan.goap_plan.readiness_score} not suitable for retrieval, current threshold is {app_env.get_acceptable_chunks_threshold_score()}")
        return RetrievalType.MINIMAL_RETRIEVAL
    
    return RetrievalType.FULL_RETRIEVAL

