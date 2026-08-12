"""LLM judge for graceful error handling."""

from evals.agent_eval_harness.llm_judge import score


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    inp = outputs.get("inputs", "")
    prompt = (
        "Does the agent handle errors gracefully? Check that it: "
        "does NOT expose raw errors, stack traces, or HTTP status codes; "
        "acknowledges failures honestly without fabricating data; "
        "provides helpful guidance. If no errors occurred, answer yes.\n\n"
        f"Question: {inp}\n"
        f"Response: {response}\n\n"
        "Reply ONLY: VERDICT: yes or no\nRATIONALE: your reason"
    )
    return score(prompt)
