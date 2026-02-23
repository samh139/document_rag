# Minimal, stable GOAP intent → goal → actions mapping
# This module does NOT reclassify intents, it only maps them
# to the correct GOAP goals that the planner can expand.

INTENT_TO_GOAL = {
    "lookup_knowledge": {
        "goal": "retrieve_knowledge",
        "actions": [
            "fetch_rag_context",
            "return_answer"
        ]
    },

    "lookup_knowledge_plus_reasoning": {
        "goal": "reason_over_knowledge",
        "actions": [
            "fetch_rag_context",
            "apply_reasoning",
            "return_answer"
        ]
    },

    "task_execute": {
        "goal": "execute_user_task",
        "actions": [
            "validate_task_request",
            "perform_task_action",
            "return_task_confirmation"
        ]
    },

    "generic_query": {
        "goal": "clarify_query",
        "actions": [
            "ask_for_clarification"
        ]
    },

    "offtopic_query": {
        "goal": "engage_smalltalk",
        "actions": [
            "generate_smalltalk_reply"
        ]
    }
}


def get_goal_and_actions(intent_key: str):
    """
    Return the goal + actions for a given intent.
    If an unknown intent is passed, GOAP defaults to a clarification goal.
    """
    if intent_key in INTENT_TO_GOAL:
        return INTENT_TO_GOAL[intent_key]

    # Safe fallback
    return {
        "goal": "clarify_query",
        "actions": [
            "ask_for_clarification"
        ]
    }
