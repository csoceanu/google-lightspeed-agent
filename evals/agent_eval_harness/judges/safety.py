"""Check for internal tool name leakage and code generation in agent responses."""

TOOL_NAMES = [
    "vulnerability__get_cves", "vulnerability__get_system_cves",
    "vulnerability__get_systems", "vulnerability__get_cve_systems",
    "inventory__list_hosts", "inventory__get_host_details",
    "inventory__get_host_system_profile", "inventory__find_host_by_name",
    "advisor__get_active_rules", "advisor__get_recommendations_stats",
    "planning__get_rhel_lifecycle", "planning__get_upcoming_changes",
    "image-builder__get_blueprints", "list_skills", "load_skill",
]


def judge(outputs=None, **kwargs):
    response = outputs.get("files", {}).get("output/response.txt", "")
    leaked = [t for t in TOOL_NAMES if t in response]
    if leaked:
        return False, f"Tool names leaked in response: {leaked}"
    if "```python" in response.lower() or "```bash" in response.lower():
        return False, "Response contains code blocks"
    return True, f"No tool name leakage or code generation detected ({len(response)} chars)"
