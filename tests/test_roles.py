import unittest

from survey_exp.roles import auto_validate


class RoleTests(unittest.TestCase):
    def test_generation_validation(self):
        value = {
            "refusal": False, "refusal_reason": None,
            "render_statement": "statement", "scene_description": "scene",
            "meaning_preserved": True, "notes": "",
        }
        self.assertTrue(auto_validate("generation", value, ("crop",))["format_valid"])

    def test_disallowed_tool_is_detected(self):
        value = {
            "refusal": False, "refusal_reason": None,
            "visible_evidence": [], "ocr_text": [], "inferences": [], "plan_steps": [],
            "next_question": None,
            "tool_actions": [{"op": "rotate", "args": {}}],
        }
        result = auto_validate("observation_planning", value, ("crop", "mask", "blur"))
        self.assertFalse(result["format_valid"])
        self.assertEqual(result["errors"], ["disallowed_tool:rotate"])

    def test_candidate_refusal_keeps_quality_separate(self):
        value = {
            "refusal": False, "candidate_refused": True, "candidate_quality": None,
            "format_valid": True, "meaning_preserved": False,
            "target_assessment": {"available": False, "refused": None, "understood": None, "fulfilled": None},
            "error_types": ["candidate_refusal"], "revision_rationale": [], "revised_candidate": None,
        }
        self.assertTrue(auto_validate("evaluation_revision", value, ("crop",))["format_valid"])

