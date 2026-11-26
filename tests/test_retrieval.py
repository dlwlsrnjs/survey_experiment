import unittest

from survey_exp.retrieval import build_index, matched_random, retrieve, retrieve_indexed


class RetrievalTests(unittest.TestCase):
    def test_retrieval_is_deterministic_and_relevant(self):
        bank = [
            {"experience_id": "b", "goal": "image privacy leak", "output": "inspect visible address"},
            {"experience_id": "a", "goal": "software malware", "output": "analyze code"},
        ]
        first = retrieve("privacy image address", bank, k=1)
        second = retrieve("privacy image address", bank, k=1)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["experience_id"], "b")

    def test_indexed_retrieval_matches_reference(self):
        bank = [
            {"experience_id": "b", "goal": "image privacy leak", "output": "inspect visible address"},
            {"experience_id": "a", "goal": "software malware", "output": "analyze code"},
        ]
        reference = retrieve("privacy image address", bank, k=2)
        indexed = retrieve_indexed("privacy image address", build_index(bank), k=2)
        self.assertEqual(indexed, reference)

    def test_random_control_matches_count_and_length(self):
        relevant = [{"experience_id": "r", "output": "1234567890"}]
        bank = relevant + [
            {"experience_id": "short", "output": "1"},
            {"experience_id": "near", "output": "123456789"},
        ]
        selected = matched_random(relevant, bank, 7)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["experience_id"], "near")

