"""Check whether the agent mentioned the expected MCP tools in its response."""


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    ann = outputs.get("annotations", {})
    expected_tools = ann.get("expected_tools", [])

    if not expected_tools:
        return True, "No tools expected for this question"

    response_lower = response.lower()
    found = []
    missed = []
    for tool in expected_tools:
        # Extract short name: vulnerability__get_cves → get_cves
        short = tool.split("__")[-1] if "__" in tool else tool
        if short.lower() in response_lower:
            found.append(short)
        else:
            missed.append(short)

    if not missed:
        return True, f"All expected tools mentioned: {found}"
    if found:
        return False, f"Partial: found {found}, missing {missed}"
    return False, f"No expected tools mentioned. Expected: {[t.split('__')[-1] for t in expected_tools]}"
