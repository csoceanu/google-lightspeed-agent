"""Validate that the agent returned a valid response."""


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    if not response or response.startswith("[ERROR]"):
        return False, f"No valid response received: {response[:100]}"
    if len(response) < 20:
        return False, f"Response too short ({len(response)} chars)"
    return True, f"Valid response received ({len(response)} chars)"
