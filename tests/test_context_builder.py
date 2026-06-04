"""Unit tests for shared.context_builder helpers introduced for the endpoint
hard-gate + agent-domain pre-processing.

Run with:
    cd erc-8004-ai-service
    ./.venv/bin/python -m unittest tests.test_context_builder -v
"""
from __future__ import annotations

import unittest

from shared.context_builder import (
    agent_domain_block,
    domain_service_names,
    endpoint_matches_services,
    is_generic_service,
)


class TestIsGenericService(unittest.TestCase):
    def test_generic_names(self):
        for n in ("web", "WEB", " oasf ", "OASF", "a2a", "email"):
            self.assertTrue(is_generic_service(n), n)

    def test_non_generic_names(self):
        for n in ("celofx", "clawnews", "sentinel8004", ""):
            # empty is not generic (it's filtered out as empty elsewhere); special
            # business names also are not generic.
            self.assertFalse(is_generic_service(n), n)


class TestDomainServiceNames(unittest.TestCase):
    def test_filters_generic_and_dedupes(self):
        services = [
            {"name": "web", "endpoint": "https://x.com"},
            {"name": "OASF", "endpoint": "https://github.com/agntcy/oasf/"},
            {"name": "celofx", "endpoint": "https://celofx.vercel.app"},
            {"name": "celofx", "endpoint": "https://celofx.vercel.app/v2"},  # dupe
            {"name": "a2a", "endpoint": "agent2agent://..."},
        ]
        self.assertEqual(domain_service_names(services), ["celofx"])

    def test_caps_at_max_names(self):
        services = [{"name": f"svc{i}", "endpoint": "https://x"} for i in range(10)]
        self.assertEqual(len(domain_service_names(services, max_names=3)), 3)

    def test_ignores_empty_name(self):
        self.assertEqual(domain_service_names([{"name": "", "endpoint": "https://x"}]), [])


class TestEndpointMatchesServices(unittest.TestCase):
    services = [
        {"name": "celofx", "endpoint": "https://celofx.vercel.app"},
        {"name": "web", "endpoint": "https://example.com"},
    ]

    def test_exact_match(self):
        self.assertTrue(endpoint_matches_services("https://celofx.vercel.app", self.services))

    def test_substring_match_extends_path(self):
        self.assertTrue(endpoint_matches_services("https://celofx.vercel.app/v1", self.services))

    def test_case_and_trailing_slash(self):
        self.assertTrue(endpoint_matches_services("HTTPS://CeloFX.vercel.app/", self.services))

    def test_no_match(self):
        self.assertFalse(endpoint_matches_services("https://unrelated.com", self.services))

    def test_empty_feedback_endpoint(self):
        self.assertFalse(endpoint_matches_services("", self.services))
        self.assertFalse(endpoint_matches_services("   ", self.services))


class TestAgentDomainBlock(unittest.TestCase):
    def test_filters_and_caps(self):
        block = agent_domain_block(
            services=[
                {"name": "web", "endpoint": "https://x"},
                {"name": "celofx", "endpoint": "https://celofx.vercel.app"},
            ],
            oasf_domains=[
                "technology/blockchain",
                "technology/artificial_intelligence",
                "technology/software_engineering/apis_integration",
                "media/news",  # capped out by domains_k=3
            ],
            oasf_skills=["natural_language_processing/text_classification"],
            tags=["defi", "forex"],
            domains_k=3,
            skills_k=5,
        )
        self.assertEqual(block["service_names"], ["celofx"])
        self.assertEqual(len(block["oasf_domains"]), 3)
        self.assertIn("technology/blockchain", block["oasf_domains"])
        self.assertEqual(block["oasf_skills"], ["natural_language_processing/text_classification"])
        self.assertEqual(block["tags"], ["defi", "forex"])


if __name__ == "__main__":
    unittest.main()
