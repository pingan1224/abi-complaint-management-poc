import json
import sqlite3
import tempfile
import unittest
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from workflow import Workflow, Service, ContractError

NORMAL = dict(summary='Customer disputes a late fee.', suggested_route='Card Billing Disputes',
              urgency='Medium', escalation=False, abstain=False, rationale='Employee must check payment records.',
              human_review_required='No', uncertainty='')
def fixture(**changes):
    return lambda narrative: ('ok', json.dumps(NORMAL | changes), {'mode': 'synthetic fixture'})

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.w = Workflow(Path(self.temp.name) / 'test.sqlite')
        self.case = dict(case_id='synthetic-1', narrative='My credit card bill includes a late fee. I paid on time.')
    def tearDown(self): self.temp.cleanup()
    def review(self, **changes):
        return self.w.review(self.case['case_id'], **(dict(reviewer_id='SCRIPTED_TEST_REVIEWER', decision='accept',
            action_id='PREPARE_BILLING_REVIEW', reason='Synthetic workflow check',
            original_checked=True, semantic_support_checked=True) | changes))
    def test_full_persistent_path(self):
        b=self.w.submit(self.case, True, fixture())
        self.assertEqual(b['queue'],'employee_review')
        self.assertFalse(self.review()['operational_execution'])
        self.assertEqual(Workflow(self.w.db).show(self.case['case_id'])['state'],'RECORDED_FOR_SIMULATED_HANDOFF')
    def test_original_saved_before_model(self):
        def service(n):
            self.assertEqual(self.w.show(self.case['case_id'])['state'],'INTAKE_SAVED')
            self.assertEqual(n,self.case['narrative'])
            return fixture()(n)
        self.w.submit(self.case,True,service)
    def test_unknown_scope_skips_model(self):
        def fail(n): raise AssertionError('must not call')
        self.assertEqual(self.w.submit(self.case,False,fail)['queue'],'manual_intake')
    def test_raw_risk_survives_failure(self):
        self.case['narrative']='I did not authorize a purchase on my card.'
        b=self.w.submit(self.case,True,lambda n: ('timeout',None,{}))
        self.assertEqual(b['queue'],'manual_triage')
        self.assertTrue(b['specialist_attention_required'])
        self.assertEqual(b['allowed_actions'],['MANUAL_REVIEW'])
    def test_risk_cannot_be_downgraded(self):
        self.case['narrative']='An unauthorized charge is on my card.'
        self.w.submit(self.case,True,fixture())
        with self.assertRaises(ContractError): self.review()
        r=self.review(decision='modify',action_id='PREPARE_SPECIALIST_REVIEW')
        self.assertTrue(r['specialist_attention_required'])
    def test_uncertainty_requires_specialist(self):
        b=self.w.submit(self.case,True,fixture(uncertainty='Timing is unclear.'))
        self.assertEqual(b['queue'],'specialist_review')
    def test_abstention_manual(self):
        b=self.w.submit(self.case,True,fixture(abstain=True))
        self.assertEqual(b['allowed_actions'],['MANUAL_REVIEW'])
    def test_malformed_and_duplicate_json(self):
        for i,raw in enumerate(('{','{"summary":1,"summary":2}')):
            self.case['case_id']=str(i)
            self.assertEqual(self.w.submit(self.case,True,lambda n: ('ok',raw,{}))['queue'],'manual_triage')
    def test_bad_boolean(self):
        self.assertEqual(self.w.submit(self.case,True,fixture(escalation='false'))['queue'],'manual_triage')
    def test_review_checks_and_replay(self):
        self.w.submit(self.case,True,fixture())
        with self.assertRaises(ContractError): self.review(original_checked=False)
        self.review()
        with self.assertRaises(ContractError): self.review()
    def test_duplicate_case_does_not_overwrite(self):
        self.w.submit(self.case,True,fixture())
        with self.assertRaises(sqlite3.IntegrityError): self.w.submit(self.case,True,fixture())
    def test_pause_blocks_handoff_and_submission(self):
        self.w.submit(self.case,True,fixture()); self.w.pause('Operator stop test')
        with self.assertRaises(ContractError): self.review()
        with self.assertRaises(ContractError): self.w.submit(dict(self.case,case_id='new'),True,fixture())
    def test_audit_failure_rolls_back_review(self):
        self.w.submit(self.case,True,fixture())
        old=self.w.event
        def broken(*args): raise sqlite3.OperationalError('injected audit failure')
        self.w.event=broken
        with self.assertRaises(sqlite3.OperationalError): self.review()
        self.w.event=old
        self.assertEqual(self.w.show(self.case['case_id'])['state'],'PENDING_EMPLOYEE_REVIEW')
    def test_extra_intake_target_rejected(self):
        with self.assertRaises(ContractError): self.w.submit(self.case | {'reference_route':'answer'},True,fixture())
    def test_http_adapter_envelope_and_model_identity(self):
        observed=[]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                req=json.loads(self.rfile.read(int(self.headers['Content-Length']))); observed.append(req)
                body=json.dumps(dict(model=req['model'],choices=[dict(message=dict(content=json.dumps(NORMAL)))])).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            def log_message(self,*args): pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        t=Thread(target=server.serve_forever,daemon=True); t.start()
        try:
            b=self.w.submit(self.case,True,Service('http://127.0.0.1:%s/v1/chat/completions'%server.server_port,'stub'))
            self.assertEqual(b['queue'],'employee_review', b['service'])
            self.assertEqual(observed[0]['messages'][1],dict(role='user',content=self.case['narrative']))
        finally: server.shutdown(); server.server_close(); t.join()

if __name__=='__main__': unittest.main(verbosity=2)
