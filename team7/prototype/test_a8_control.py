"""Synthetic adversarial control checks; model replies are fixtures, not inference."""
import json
import sqlite3
import unittest
from test_workflow import WorkflowTests, fixture
from workflow import screen, SCREEN_VERSION, ContractError

class A8ControlTests(WorkflowTests):
    def test_paraphrases_override_routine_advice(self):
        cases = ["I never authorized this charge.", "I didn’t authorize this charge.",
                 "Someone used my card without my permission.",
                 "I did not make this purchase.", "I NEVER   AUTHORIZED this charge."]
        for i, text in enumerate(cases):
            with self.subTest(text=text):
                self.case = dict(case_id='paraphrase-'+str(i), narrative=text)
                b = self.w.submit(self.case, True, fixture())
                self.assertEqual(b['queue'], 'specialist_review')
                self.assertTrue(b['specialist_attention_required'])
                self.assertEqual(b['screen_version'], SCREEN_VERSION)
                with self.assertRaises(ContractError): self.review()
                self.assertEqual(self.w.show(self.case['case_id'])['state'], 'PENDING_EMPLOYEE_REVIEW')
    def test_paraphrase_survives_timeout_and_invalid_output(self):
        for i, result in enumerate([('timeout', None, {}), ('ok', '{', {})]):
            self.case = dict(case_id='failure-'+str(i), narrative='I never authorized this charge.')
            b = self.w.submit(self.case, True, lambda n: result)
            self.assertEqual(b['queue'], 'manual_triage')
            self.assertTrue(b['specialist_attention_required'])
            self.assertEqual(b['allowed_actions'], ['MANUAL_REVIEW'])
    def test_mixed_claim_does_not_remove_risk(self):
        self.assertTrue(screen('The first charge was authorized. I never authorized the second.'))
    def test_negated_legacy_phrase_remains_conservative(self):
        # Known false positive retained deliberately, not claimed as semantic understanding.
        self.assertTrue(screen('This is not an unauthorized charge; I only dispute the fee.'))
    def test_plain_fee_has_no_independent_flag(self):
        self.assertEqual(screen('I authorized the purchase but dispute the late fee.'), ())
    def test_unknown_paraphrase_exposes_limit(self):
        # Explicit evidence that this finite phrase control is incomplete.
        self.assertEqual(screen('Somebody else bought this using my card.'), ())
    def test_risk_signal_and_version_are_durable(self):
        self.case['narrative'] = 'I never authorized this charge.'
        self.w.submit(self.case, True, fixture())
        with sqlite3.connect(self.w.db) as c:
            saved = json.loads(c.execute("SELECT body FROM events WHERE kind='REVIEW_PREPARED'").fetchone()[0])
        self.assertEqual(saved['screen_version'], SCREEN_VERSION)
        self.assertTrue(any('raw_v2:never_authorized' in r for r in saved['guard']['reasons']))

if __name__ == '__main__':
    # Avoid counting the inherited Team 7 tests twice in this focused suite.
    suite = unittest.TestSuite(A8ControlTests(n) for n in A8ControlTests.__dict__ if n.startswith('test_'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
