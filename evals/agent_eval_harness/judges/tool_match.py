"""Check whether the agent invoked the expected MCP tools.

Checks the full A2A execution trace (a2a_response.json) first, falling
back to the user-facing response text (response.txt) when the trace is
not available.  This ensures tool calls that appear in the execution
trace but are not echoed in the final response are still detected.
"""

import json


def judge(outputs=None, **kwargs):
    ann = outputs.get("annotations", {})
    expected_tools = ann.get("expected_tools", [])

    if not expected_tools:
        return True, "No tools expected for this question"

    # Prefer the full A2A execution trace; fall back to user-facing text.
    files = outputs.get("files", {})
    a2a_raw = files.get("output/a2a_response.json", "")
    response_txt = files.get("output/response.txt", "")

    if a2a_raw:
        try:
            parsed = json.loads(a2a_raw)
            search_text = json.dumps(parsed).lower()
        except (json.JSONDecodeError, TypeError):
            search_text = a2a_raw.lower()
    elif response_txt:
        search_text = response_txt.lower()
    else:
        return False, "No response data available to check tool usage"

    found = []
    missed = []
    for tool in expected_tools:
        short = tool.split("__")[-1] if "__" in tool else tool
        if short.lower() in search_text:
            found.append(short)
        else:
            missed.append(short)

    if not missed:
        return True, f"All expected tools found in trace: {found}"
    if found:
        return False, f"Partial: found {found}, missing {missed}"
    return False, (
        f"No expected tools found in trace. "
        f"Expected: {[t.split('__')[-1] for t in expected_tools]}"
    )
