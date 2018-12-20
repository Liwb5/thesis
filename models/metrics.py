from rouge import Rouge

def rouge_metric(hyp, ref):
    """
    Calculate rouge metric.

    Args:
        hyp (list): a list of tokens that is predicted by the model
        ref (list): a list of tokens that is the reference
    """
    rouge = Rouge()
    # if avg=False, returns a list of n dicts.
    # if avg=True, returns a single dict with average values.
    scores = rouge.get_scores(hyp, ref, avg=True) 
    return scores  #{"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-3": { ... }}

