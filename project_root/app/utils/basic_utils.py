from typing import List, Dict

def normalize_scores(items: List[Dict], score_key: str) -> None:
    """
    Normalize scores in-place between 0 and 1 for the given key.
    """
    if not items:
        return
    
    scores = [item.get(score_key, 0) for item in items]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero
    if max_score == min_score:
        for item in items:
            item[score_key] = 1.0  # if all scores are equal, set to 1
    else:
        for item in items:
            item[score_key] = (item.get(score_key, 0) - min_score) / (max_score - min_score)