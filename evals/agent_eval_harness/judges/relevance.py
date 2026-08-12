"""LLM judge for response relevance to the question."""

from evals.agent_eval_harness.llm_judge import score


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    inp = outputs.get("inputs", "")
    prompt = (
        "Does the response directly address the user's question? "
        "If the agent encountered an error, a relevant response explains "
        "what happened and suggests next steps.\n\n"
        f"Question: {inp}\n"
        f"Response: {response}\n\n"
        "Reply ONLY: VERDICT: yes or no\nRATIONALE: your reason"
    )
    return score(prompt)
