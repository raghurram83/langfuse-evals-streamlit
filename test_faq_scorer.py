import unittest

from app import FAQScorer


class TestFAQScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = FAQScorer()

    def test_high_quality_accept(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "ACCEPT")
        self.assertGreaterEqual(result["overall_faq_score"], 0.75)

    def test_incomplete_answer_review(self):
        faq = {
            "question_confidence": 0.8,
            "answer_confidence": 0.55,
            "relevance_score": 0.7,
            "completeness_score": 0.5,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REVIEW")

    def test_high_relevance_low_depth_reject(self):
        faq = {
            "question_confidence": 0.8,
            "answer_confidence": 0.3,
            "relevance_score": 0.8,
            "completeness_score": 0.3,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")

    def test_duplicate_penalty(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.9,  # major penalty
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertTrue(result["overall_faq_score"] < 0.75)

    def test_missing_pii_reject(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.1,
            "pii_removed": False,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")
        self.assertEqual(result["overall_faq_score"], 0.0)

    def test_semantic_mismatch(self):
        faq = {
            "question_confidence": 0.6,
            "answer_confidence": 0.6,
            "relevance_score": 0.2,
            "completeness_score": 0.5,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")

    def test_strong_answer_poor_question(self):
        faq = {
            "question_confidence": 0.4,
            "answer_confidence": 0.9,
            "relevance_score": 0.8,
            "completeness_score": 0.8,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertIn(result["decision"], ["REVIEW", "REJECT"])

    def test_multistep_clarity_accept(self):
        faq = {
            "question_confidence": 0.8,
            "answer_confidence": 0.9,
            "relevance_score": 0.85,
            "completeness_score": 0.85,
            "redundancy_score": 0.1,
            "pii_removed": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "ACCEPT")

    def test_reject_agent_name(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.1,
            "pii_removed": True,
            "name_flag": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")

    def test_reject_routing(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.1,
            "pii_removed": True,
            "routing_flag": True,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")

    def test_reject_low_worthiness(self):
        faq = {
            "question_confidence": 0.9,
            "answer_confidence": 0.9,
            "relevance_score": 0.9,
            "completeness_score": 0.9,
            "redundancy_score": 0.1,
            "pii_removed": True,
            "faq_worthiness": 0.2,
        }
        result = self.scorer.score_faq(faq)
        self.assertEqual(result["decision"], "REJECT")


if __name__ == "__main__":
    unittest.main()
