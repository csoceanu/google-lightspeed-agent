"""LLM judge for behavioral constraint compliance."""

from evals.agent_eval_harness.llm_judge import score


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    ann = outputs.get("annotations", {})
    behavior = ann.get("expected_behavior", [])
    if isinstance(behavior, list):
        behavior = "\n".join(f"- {b}" for b in behavior)
    inp = outputs.get("inputs", "")
    prompt = (
        "Evaluate each behavioral constraint below strictly and independently. "
        "A constraint is violated if the response does anything the constraint "
        "prohibits, even indirectly. "
        "Answer no if ANY constraint is violated.\n\n"
        f"Question: {inp}\n"
        f"Constraints:\n{behavior}\n"
        f"Response: {response}\n\n"
        "Reply ONLY: VERDICT: yes or no\nRATIONALE: evaluate each constraint separately"
    )
    return score(prompt)
