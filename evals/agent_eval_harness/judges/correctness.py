"""LLM judge for factual correctness against expected response."""

from evals.agent_eval_harness.llm_judge import score


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    ann = outputs.get("annotations", {})
    expected = ann.get("expected_response", "")
    inp = outputs.get("inputs", "")
    prompt = (
        "Does the agent's response contain the key facts from the expected "
        "response? If the agent encountered an error (e.g. 403), evaluate "
        "whether the error handling was appropriate.\n\n"
        f"Question: {inp}\n"
        f"Expected: {expected}\n"
        f"Response: {response}\n\n"
        "Reply ONLY: VERDICT: yes or no\nRATIONALE: your reason"
    )
    return score(prompt)
