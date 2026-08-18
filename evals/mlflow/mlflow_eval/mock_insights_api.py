#!/usr/bin/env python3
"""
Mock Red Hat Insights API server backed by mock_mcp_data.json.

Mimics console.redhat.com API responses so the real MCP server can
run against synthetic data without Red Hat SSO credentials or VPN.

Point the MCP server at this mock:
    INSIGHTS_BASE_URL=http://localhost:9000 insights-mcp http

The mock serves the same API paths the MCP server calls, returning
data from mock_mcp_data.json. No authentication is required.

Routes match the exact paths each MCP server calls:
  - /api/vulnerability/v1/vulnerabilities/cves
  - /api/vulnerability/v1/cves/{id}
  - /api/vulnerability/v1/cves/{id}/affected_systems
  - /api/vulnerability/v1/systems/{uuid}/cves
  - /api/vulnerability/v1/systems
  - /api/inventory/v1/hosts
  - /api/inventory/v1/hosts/{ids}
  - /api/inventory/v1/hosts/{ids}/system_profile
  - /api/inventory/v1/hosts/{ids}/tags
  - /api/insights/v1/rule/
  - /api/insights/v1/rule/{rule_id}/
  - /api/insights/v1/rule/{rule_id}/systems/
  - /api/insights/v1/rule/{rule_id}/systems_detail/
  - /api/insights/v1/kcs/{node_id}/
  - /api/insights/v1/stats/rules/
  - /api/roadmap/v1/lifecycle/rhel
  - /api/roadmap/v1/lifecycle/app-streams/{major}
  - /api/roadmap/v1/lifecycle/app-streams/streams
  - /api/roadmap/v1/upcoming-changes
  - /api/roadmap/v1/relevant/upcoming-changes
  - /api/roadmap/v1/relevant/lifecycle/rhel
  - /api/roadmap/v1/relevant/lifecycle/app-streams
  - /api/content-sources/v1.0/repositories/
  - /api/rbac/v1/access/
  - /api/rhsm/v2/activation_keys
  - /api/rhsm/v2/activation_keys/{name}
  - /api/remediations/v1/resolutions (POST)
  - /api/remediations/v1/remediations (POST)
  - /api/remediations/v1/remediations/{id}/playbook (GET)
  - /api/vmaas/v3/vulnerabilities (POST)
  - /api/image-builder/v1/*

Usage:
    python -m mlflow_eval.mock_insights_api [--port 9000]
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger("mock_insights_api")

MOCK_DATA_PATH = Path(__file__).resolve().parent.parent / "mock_mcp_data.json"


def load_mock_data(path: str | None = None) -> dict:
    p = Path(path) if path else MOCK_DATA_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class MockInsightsHandler(BaseHTTPRequestHandler):
    mock_data: dict = {}

    # ── HTTP methods ──────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        # Split path into segments for precise matching
        parts = [unquote(p) for p in path.split("/") if p]

        result = self._route_get(parts, params)
        if result is not None:
            # Playbook endpoint returns raw YAML text
            if isinstance(result, str) and not result.startswith("{") and not result.startswith("["):
                self._text_response(200, result)
            else:
                self._json_response(200, result)
        else:
            logger.warning("Unhandled GET path: %s", path)
            self._json_response(200, {"data": [], "meta": {"count": 0, "total": 0}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        parts = [unquote(p) for p in path.split("/") if p]

        result = self._route_post(parts, body)
        self._json_response(200, result)

    # ── GET routing ───────────────────────────────────────────────

    def _route_get(self, parts: list[str], params: dict) -> dict | list | str | None:
        """Route GET requests by matching path segments exactly."""

        # ── Vulnerability: /api/vulnerability/v1/* ────────────────
        if _matches(parts, ["api", "vulnerability", "v1"]):
            rest = parts[3:]

            # GET /api/vulnerability/v1/vulnerabilities/cves
            if rest == ["vulnerabilities", "cves"]:
                return self._vuln_list_cves(params)

            # GET /api/vulnerability/v1/cves/{id}/affected_systems
            if len(rest) == 3 and rest[0] == "cves" and rest[2] == "affected_systems":
                return self._vuln_affected_systems(rest[1], params)

            # GET /api/vulnerability/v1/cves/{id}
            if len(rest) == 2 and rest[0] == "cves":
                return self._vuln_get_cve(rest[1])

            # GET /api/vulnerability/v1/systems/{uuid}/cves
            if len(rest) == 3 and rest[0] == "systems" and rest[2] == "cves":
                return self._vuln_system_cves(rest[1])

            # GET /api/vulnerability/v1/systems
            if rest == ["systems"]:
                return self._vuln_list_systems()

        # ── Inventory: /api/inventory/v1/* ────────────────────────
        if _matches(parts, ["api", "inventory", "v1"]):
            rest = parts[3:]

            # GET /api/inventory/v1/hosts/{ids}/system_profile
            if len(rest) == 3 and rest[0] == "hosts" and rest[2] == "system_profile":
                return self._inv_system_profile(rest[1])

            # GET /api/inventory/v1/hosts/{ids}/tags
            if len(rest) == 3 and rest[0] == "hosts" and rest[2] == "tags":
                return self._inv_host_tags(rest[1])

            # GET /api/inventory/v1/hosts/{ids}
            if len(rest) == 2 and rest[0] == "hosts":
                return self._inv_host_details(rest[1])

            # GET /api/inventory/v1/hosts
            if rest == ["hosts"]:
                return self._inv_list_hosts(params)

        # ── Advisor: /api/insights/v1/* ───────────────────────────
        if _matches(parts, ["api", "insights", "v1"]):
            rest = parts[3:]

            # GET /api/insights/v1/stats/rules
            if rest == ["stats", "rules"]:
                return self._advisor_stats()

            # GET /api/insights/v1/kcs/{node_id}
            if len(rest) == 2 and rest[0] == "kcs":
                return self._advisor_rule_by_node(rest[1])

            # GET /api/insights/v1/rule/{rule_id}/systems_detail
            if len(rest) == 3 and rest[0] == "rule" and rest[2] == "systems_detail":
                return self._advisor_hosts_details(rest[1])

            # GET /api/insights/v1/rule/{rule_id}/systems
            if len(rest) == 3 and rest[0] == "rule" and rest[2] == "systems":
                return self._advisor_hosts_hitting(rest[1])

            # GET /api/insights/v1/rule/{rule_id}
            if len(rest) == 2 and rest[0] == "rule":
                return self._advisor_rule_detail(rest[1])

            # GET /api/insights/v1/rule
            if rest == ["rule"]:
                return self._advisor_list_rules(params)

        # ── Planning: /api/roadmap/v1/* ───────────────────────────
        if _matches(parts, ["api", "roadmap", "v1"]):
            rest = parts[3:]

            # Relevant endpoints (must check before non-relevant)
            # GET /api/roadmap/v1/relevant/upcoming-changes
            if rest == ["relevant", "upcoming-changes"]:
                return self._planning_relevant_upcoming(params)

            # GET /api/roadmap/v1/relevant/lifecycle/rhel
            if rest == ["relevant", "lifecycle", "rhel"]:
                return self._planning_relevant_rhel_lifecycle(params)

            # GET /api/roadmap/v1/relevant/lifecycle/app-streams
            if rest == ["relevant", "lifecycle", "app-streams"]:
                return self._planning_relevant_appstreams(params)

            # GET /api/roadmap/v1/lifecycle/rhel
            if rest == ["lifecycle", "rhel"]:
                return self._planning_rhel_lifecycle()

            # GET /api/roadmap/v1/lifecycle/app-streams/streams
            if rest == ["lifecycle", "app-streams", "streams"]:
                return self._planning_appstreams_streams(params)

            # GET /api/roadmap/v1/lifecycle/app-streams/{major}
            if len(rest) == 3 and rest[0] == "lifecycle" and rest[1] == "app-streams":
                return self._planning_appstreams_raw(rest[2], params)

            # GET /api/roadmap/v1/upcoming-changes
            if rest == ["upcoming-changes"]:
                return self._planning_upcoming_changes()

        # ── Image Builder: /api/image-builder/v1/* ────────────────
        if _matches(parts, ["api", "image-builder", "v1"]):
            return self._image_builder(parts)

        # ── Content Sources: /api/content-sources/v1.0/* ──────────
        if _matches(parts, ["api", "content-sources", "v1.0"]):
            rest = parts[3:]
            if rest == ["repositories"] or (len(rest) == 1 and rest[0] == "repositories"):
                return self._content_sources(params)

        # ── RBAC: /api/rbac/v1/access ─────────────────────────────
        if _matches(parts, ["api", "rbac", "v1"]):
            rest = parts[3:]
            if rest == ["access"]:
                return self._rbac_access(params)

        # ── RHSM: /api/rhsm/v2/* ─────────────────────────────────
        if _matches(parts, ["api", "rhsm", "v2"]):
            rest = parts[3:]
            if len(rest) == 2 and rest[0] == "activation_keys":
                return self._rhsm_key_detail(rest[1])
            if rest == ["activation_keys"]:
                return self._rhsm_list_keys(params)

        # ── Remediations GET: /api/remediations/v1/* ──────────────
        if _matches(parts, ["api", "remediations", "v1"]):
            rest = parts[3:]
            # GET /api/remediations/v1/remediations/{id}/playbook
            if len(rest) == 3 and rest[0] == "remediations" and rest[2] == "playbook":
                return self._remediations_playbook(rest[1])

        # ── VMaaS GET fallback: /api/vmaas/v3/* ───────────────────
        if _matches(parts, ["api", "vmaas", "v3"]):
            return {"cve_list": [], "manually_fixable_cve_list": [], "unpatched_cve_list": []}

        return None

    # ── POST routing ──────────────────────────────────────────────

    def _route_post(self, parts: list[str], body: dict) -> dict:
        """Route POST requests."""

        # POST /api/vmaas/v3/vulnerabilities
        if _matches(parts, ["api", "vmaas", "v3", "vulnerabilities"]):
            return self._vmaas_vulnerabilities(body)

        # POST /api/remediations/v1/resolutions
        if _matches(parts, ["api", "remediations", "v1", "resolutions"]):
            return self._remediations_resolutions(body)

        # POST /api/remediations/v1/remediations
        if _matches(parts, ["api", "remediations", "v1", "remediations"]):
            return self._remediations_create(body)

        logger.warning("Unhandled POST path: %s", "/".join(parts))
        return {"id": "mock-result"}

    # ── Vulnerability helpers ─────────────────────────────────────

    def _vuln_list_cves(self, params):
        """GET /api/vulnerability/v1/vulnerabilities/cves"""
        cves = list(self.mock_data.get("vulnerability", {}).get("cves", {}).values())
        return {
            "data": cves,
            "meta": {"total_items": len(cves), "limit": 20, "offset": 0},
        }

    def _vuln_get_cve(self, cve_id):
        """GET /api/vulnerability/v1/cves/{cve_id}"""
        cve = self.mock_data.get("vulnerability", {}).get("cves", {}).get(cve_id.upper())
        if cve:
            return {"data": cve}
        return {"errors": [{"status": 404, "detail": f"{cve_id} not found"}]}

    def _vuln_affected_systems(self, cve_id, params):
        """GET /api/vulnerability/v1/cves/{cve_id}/affected_systems"""
        vuln = self.mock_data.get("vulnerability", {})
        system_ids = vuln.get("cve_to_systems", {}).get(cve_id.upper(), [])
        all_systems = vuln.get("systems", [])
        cve_data = vuln.get("cves", {}).get(cve_id.upper(), {})
        advisory = (cve_data.get("attributes", {}).get("advisories_list") or [None])[0]

        # Filter by display_name if filter param is present
        name_filter = params.get("filter", [None])[0]

        data = []
        for sys in all_systems:
            if sys["id"] in system_ids:
                # Build affected_system response object
                entry = {
                    "id": sys["id"],
                    "type": "system",
                    "attributes": {
                        "display_name": sys["attributes"]["display_name"],
                        "os": sys["attributes"]["os"],
                        "updated": sys["attributes"]["updated"],
                        "status": "Affected",
                        "advisory": advisory,
                        "rule": None,
                    },
                }
                if name_filter:
                    if name_filter.lower() in entry["attributes"]["display_name"].lower():
                        data.append(entry)
                else:
                    data.append(entry)

        return {
            "data": data,
            "meta": {"total_items": len(data), "limit": 20, "offset": 0},
        }

    def _vuln_system_cves(self, sys_id):
        """GET /api/vulnerability/v1/systems/{system_uuid}/cves"""
        vuln = self.mock_data.get("vulnerability", {})
        cve_ids = vuln.get("system_to_cves", {}).get(sys_id, [])
        cves_data = vuln.get("cves", {})
        data = [cves_data[c] for c in cve_ids if c in cves_data]
        return {
            "data": data,
            "meta": {"total_items": len(data), "limit": 20, "offset": 0},
        }

    def _vuln_list_systems(self):
        """GET /api/vulnerability/v1/systems"""
        systems = self.mock_data.get("vulnerability", {}).get("systems", [])
        return {
            "data": systems,
            "meta": {"total_items": len(systems), "limit": 20, "offset": 0},
        }

    # ── Inventory helpers ─────────────────────────────────────────

    def _inv_list_hosts(self, params):
        """GET /api/inventory/v1/hosts"""
        hosts = self.mock_data.get("inventory", {}).get("hosts", [])
        hostname = params.get("hostname_or_id", [None])[0]
        per_page = int(params.get("per_page", [50])[0])

        if hostname:
            hosts = [
                h for h in hosts
                if hostname.lower() in h.get("display_name", "").lower()
                or hostname.lower() in h.get("fqdn", "").lower()
                or hostname.lower() == h.get("id", "").lower()
            ]

        hosts = hosts[:per_page]
        return {
            "total": len(hosts),
            "count": len(hosts),
            "page": 1,
            "per_page": per_page,
            "results": hosts,
        }

    def _inv_host_details(self, host_ids_str):
        """GET /api/inventory/v1/hosts/{host_ids}"""
        all_hosts = self.mock_data.get("inventory", {}).get("hosts", [])
        ids = [h.strip() for h in host_ids_str.split(",")]
        matched = [h for h in all_hosts if h["id"] in ids]
        return {
            "total": len(matched),
            "count": len(matched),
            "page": 1,
            "per_page": 50,
            "results": matched,
        }

    def _inv_system_profile(self, host_ids_str):
        """GET /api/inventory/v1/hosts/{host_ids}/system_profile"""
        all_hosts = self.mock_data.get("inventory", {}).get("hosts", [])
        ids = [h.strip() for h in host_ids_str.split(",")]
        results = []
        for h in all_hosts:
            if h["id"] in ids:
                results.append({
                    "id": h["id"],
                    "system_profile": h.get("system_profile", {}),
                    "rhsm": h.get("system_profile", {}).get("rhsm", {}),
                })
        return {
            "total": len(results),
            "count": len(results),
            "page": 1,
            "per_page": 50,
            "results": results,
        }

    def _inv_host_tags(self, host_ids_str):
        """GET /api/inventory/v1/hosts/{host_ids}/tags"""
        all_hosts = self.mock_data.get("inventory", {}).get("hosts", [])
        ids = [h.strip() for h in host_ids_str.split(",")]
        results = {}
        for h in all_hosts:
            if h["id"] in ids:
                results[h["id"]] = h.get("tags", [])
        return {
            "total": len(results),
            "count": len(results),
            "page": 1,
            "per_page": 50,
            "results": results,
        }

    # ── Advisor helpers ───────────────────────────────────────────

    def _advisor_stats(self):
        """GET /api/insights/v1/stats/rules/"""
        return self.mock_data.get("advisor", {}).get("recommendations_stats", {})

    def _advisor_list_rules(self, params):
        """GET /api/insights/v1/rule/"""
        rules = self.mock_data.get("advisor", {}).get("active_rules", [])
        text_search = params.get("text", [None])[0]
        if text_search:
            text_search = text_search.lower()
            rules = [
                r for r in rules
                if text_search in r.get("description", "").lower()
                or text_search in r.get("rule_id", "").lower()
                or text_search in r.get("summary", "").lower()
                or text_search in r.get("tags", "").lower()
            ]
        return {
            "meta": {"count": len(rules)},
            "links": {
                "first": "/api/insights/v1/rule/?limit=20&offset=0",
                "next": None,
                "previous": None,
                "last": "/api/insights/v1/rule/?limit=20&offset=0",
            },
            "data": rules,
        }

    def _advisor_rule_detail(self, rule_id):
        """GET /api/insights/v1/rule/{rule_id}/
        Returns the rule object directly (no data wrapper)."""
        for r in self.mock_data.get("advisor", {}).get("active_rules", []):
            if r["rule_id"] == rule_id:
                return r
        return {}

    def _advisor_rule_by_node(self, node_id):
        """GET /api/insights/v1/kcs/{node_id}/
        Returns a flat list of recommendation URL strings."""
        urls = []
        for r in self.mock_data.get("advisor", {}).get("active_rules", []):
            if str(r.get("node_id")) == str(node_id):
                urls.append(
                    f"console.redhat.com/insights/advisor/recommendations/{r['rule_id']}"
                )
        return urls

    def _advisor_hosts_hitting(self, rule_id):
        """GET /api/insights/v1/rule/{rule_id}/systems/
        Returns {"host_ids": [...]}."""
        hosts_map = self.mock_data.get("advisor", {}).get("hosts_hitting_rules", {})
        host_ids = hosts_map.get(rule_id, [])
        return {"host_ids": host_ids}

    def _advisor_hosts_details(self, rule_id):
        """GET /api/insights/v1/rule/{rule_id}/systems_detail/"""
        details_map = self.mock_data.get("advisor", {}).get("hosts_details_for_rules", {})
        data = details_map.get(rule_id, [])
        return {
            "meta": {"count": len(data)},
            "links": {
                "first": f"/api/insights/v1/rule/{rule_id}/systems_detail/?limit=10&offset=0",
                "next": None,
                "previous": None,
                "last": f"/api/insights/v1/rule/{rule_id}/systems_detail/?limit=10&offset=0",
            },
            "data": data,
        }

    # ── Planning helpers ──────────────────────────────────────────

    def _planning_rhel_lifecycle(self):
        """GET /api/roadmap/v1/lifecycle/rhel"""
        data = self.mock_data.get("planning", {}).get("rhel_lifecycle", [])
        return {"data": data}

    def _planning_appstreams_raw(self, major, params):
        """GET /api/roadmap/v1/lifecycle/app-streams/{major}"""
        raw = self.mock_data.get("planning", {}).get("appstreams_raw", {})
        result = raw.get(str(major))
        if result:
            return self._filter_appstreams(result, params)
        # Return empty if the major version has no data
        return {"meta": {"count": 0, "total": 0}, "data": []}

    def _planning_appstreams_streams(self, params):
        """GET /api/roadmap/v1/lifecycle/app-streams/streams"""
        result = self.mock_data.get("planning", {}).get("appstreams_streams", {})
        if result:
            return self._filter_appstreams(result, params)
        return {"meta": {"count": 0, "total": 0}, "data": []}

    def _filter_appstreams(self, result, params):
        """Apply optional query param filters to appstreams data."""
        data = list(result.get("data", []))
        name = params.get("name", [None])[0]
        stream_name = params.get("application_stream_name", [None])[0]
        stream_type = params.get("application_stream_type", [None])[0]
        kind = params.get("kind", [None])[0]

        if name:
            data = [d for d in data if d.get("name") == name]
        if stream_name:
            data = [d for d in data if d.get("application_stream_name") == stream_name]
        if stream_type:
            data = [d for d in data if d.get("application_stream_type") == stream_type]

        return {"meta": {"count": len(data), "total": len(data)}, "data": data}

    def _planning_upcoming_changes(self):
        """GET /api/roadmap/v1/upcoming-changes"""
        return self.mock_data.get("planning", {}).get("upcoming_changes", {"meta": {"count": 0, "total": 0}, "data": []})

    def _planning_relevant_upcoming(self, params):
        """GET /api/roadmap/v1/relevant/upcoming-changes"""
        return self.mock_data.get("planning", {}).get("relevant_upcoming", {"meta": {"count": 0, "total": 0}, "data": []})

    def _planning_relevant_rhel_lifecycle(self, params):
        """GET /api/roadmap/v1/relevant/lifecycle/rhel"""
        return self.mock_data.get("planning", {}).get("relevant_rhel_lifecycle", {"meta": {"count": 0, "total": 0}, "data": []})

    def _planning_relevant_appstreams(self, params):
        """GET /api/roadmap/v1/relevant/lifecycle/app-streams"""
        return self.mock_data.get("planning", {}).get("relevant_appstreams", {"meta": {"count": 0, "total": 0}, "data": []})

    # ── Image Builder helpers ─────────────────────────────────────

    def _image_builder(self, parts):
        """Handle /api/image-builder/v1/* routes."""
        ib = self.mock_data.get("image_builder", {})
        rest = parts[3:]  # after api/image-builder/v1

        if "distributions" in rest:
            return {"data": ib.get("distributions", [])}
        if "composes" in rest:
            if len(rest) >= 2:
                compose_id = rest[1]
                for c in ib.get("composes", []):
                    if c["id"] == compose_id:
                        return {"data": c}
            composes = ib.get("composes", [])
            return {"data": composes, "meta": {"count": len(composes)}}
        if "blueprints" in rest:
            if len(rest) >= 2:
                bp_id = rest[1]
                for bp in ib.get("blueprints", []):
                    if bp["id"] == bp_id or bp["name"] == bp_id:
                        return {"data": bp}
            blueprints = ib.get("blueprints", [])
            return {"data": blueprints, "meta": {"count": len(blueprints)}}
        if "openapi" in rest:
            return {"openapi": "3.0.0", "info": {"title": "Image Builder", "version": "1.0"}}
        return {"data": []}

    # ── Content Sources ───────────────────────────────────────────

    def _content_sources(self, params):
        """GET /api/content-sources/v1.0/repositories/"""
        repos = self.mock_data.get("content_sources", {}).get("repositories", [])

        # Apply optional filters
        name = params.get("name", [None])[0]
        origin = params.get("origin", [None])[0]
        content_type = params.get("content_type", [None])[0]
        arch = params.get("arch", [None])[0]
        version = params.get("version", [None])[0]

        if name:
            repos = [r for r in repos if name.lower() in r.get("name", "").lower()]
        if origin:
            repos = [r for r in repos if r.get("origin") == origin]
        if content_type:
            repos = [r for r in repos if r.get("content_type") == content_type]
        if arch:
            repos = [r for r in repos if r.get("distribution_arch") == arch]
        if version:
            repos = [r for r in repos if version in r.get("distribution_versions", [])]

        limit = int(params.get("limit", [10])[0])
        offset = int(params.get("offset", [0])[0])
        total = len(repos)
        repos = repos[offset : offset + limit]

        return {
            "data": repos,
            "meta": {"count": len(repos), "limit": limit, "offset": offset, "total": total},
        }

    # ── RBAC ──────────────────────────────────────────────────────

    def _rbac_access(self, params):
        """GET /api/rbac/v1/access/"""
        access = self.mock_data.get("rbac", {}).get("access", [])
        limit = int(params.get("limit", [20])[0])
        offset = int(params.get("offset", [0])[0])
        total = len(access)
        page = access[offset : offset + limit]
        return {
            "meta": {"count": total, "limit": limit, "offset": offset},
            "links": {
                "first": f"/api/rbac/v1/access/?limit={limit}&offset=0",
                "next": None,
                "previous": None,
                "last": f"/api/rbac/v1/access/?limit={limit}&offset=0",
            },
            "data": page,
        }

    # ── RHSM ──────────────────────────────────────────────────────

    def _rhsm_list_keys(self, params):
        """GET /api/rhsm/v2/activation_keys"""
        keys = self.mock_data.get("rhsm", {}).get("activation_keys", [])
        return {"body": keys}

    def _rhsm_key_detail(self, name):
        """GET /api/rhsm/v2/activation_keys/{name}"""
        for k in self.mock_data.get("rhsm", {}).get("activation_keys", []):
            if k["name"] == name:
                return k
        return {}

    # ── Remediations ──────────────────────────────────────────────

    def _remediations_resolutions(self, body):
        """POST /api/remediations/v1/resolutions
        Returns resolution data keyed by issue ID."""
        stored = self.mock_data.get("remediations", {}).get("resolutions", {})
        result = {}
        for issue in body.get("issues", []):
            if issue in stored:
                result[issue] = stored[issue]
            else:
                # Generate a generic resolution for unknown issues
                result[issue] = {
                    "id": issue,
                    "resolutions": [
                        {
                            "id": "fix",
                            "description": f"Apply fix for {issue}",
                            "needs_reboot": False,
                            "resolution_risk": 3,
                        }
                    ],
                }
        return result

    def _remediations_create(self, body):
        """POST /api/remediations/v1/remediations
        Returns {"id": "..."}."""
        return {"id": "pb-2024-001"}

    def _remediations_playbook(self, remediation_id):
        """GET /api/remediations/v1/remediations/{id}/playbook
        Returns raw YAML string."""
        return self.mock_data.get("remediations", {}).get(
            "playbook_yaml",
            "---\n- name: Mock playbook\n  hosts: all\n  tasks: []\n",
        )

    # ── VMaaS ─────────────────────────────────────────────────────

    def _vmaas_vulnerabilities(self, body):
        """POST /api/vmaas/v3/vulnerabilities"""
        vmaas = self.mock_data.get("vmaas", {})
        return {
            "cve_list": vmaas.get("cve_list", []),
            "manually_fixable_cve_list": vmaas.get("manually_fixable_cve_list", []),
            "unpatched_cve_list": vmaas.get("unpatched_cve_list", []),
        }

    # ── Utilities ─────────────────────────────────────────────────

    def _json_response(self, status: int, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _text_response(self, status: int, text: str):
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.debug(fmt, *args)


def _matches(parts: list[str], prefix: list[str]) -> bool:
    """Check if parts starts with the given prefix segments."""
    return len(parts) >= len(prefix) and parts[: len(prefix)] == prefix


def start_mock_api(port: int = 9000, data_path: str | None = None) -> HTTPServer:
    """Start the mock API server in a background thread. Returns the server."""
    import threading

    MockInsightsHandler.mock_data = load_mock_data(data_path)
    server = HTTPServer(("127.0.0.1", port), MockInsightsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("Mock Insights API on http://127.0.0.1:%d", port)
    return server


def main():
    parser = argparse.ArgumentParser(description="Mock Red Hat Insights API server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--data", default=None, help="Path to mock_mcp_data.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    MockInsightsHandler.mock_data = load_mock_data(args.data)
    server = HTTPServer(("127.0.0.1", args.port), MockInsightsHandler)
    logger.info("Mock Insights API on http://127.0.0.1:%d", args.port)
    logger.info(
        "Set INSIGHTS_BASE_URL=http://127.0.0.1:%d to use with MCP server",
        args.port,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
