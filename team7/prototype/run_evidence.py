"""Synthetic end-to-end run. Review steps use a clearly identified test actor."""
import argparse
import hashlib
import json
import os
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from workflow import Service, Workflow

CASES = [
    ('routine_fee','My credit card statement includes a late fee. I paid on time and would like the charge reviewed.'),
    ('unauthorized_charge','My credit card bill includes a purchase I did not authorize. I still have my card and do not recognize the merchant.'),
    ('unclear_dispute','Something on my credit card bill seems wrong, but I cannot identify the charge or date yet.')]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--endpoint',required=True); p.add_argument('--model',required=True)
    args=p.parse_args(); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    db=out/'workflow.sqlite'
    if db.exists(): raise RuntimeError('Fresh evidence directory required; no overwrite')
    w=Workflow(db); service=Service(args.endpoint,args.model)
    evidence=dict(run_at_utc=datetime.now(timezone.utc).isoformat(),hostname=platform.node(),
        python=platform.python_version(),slurm_job=os.environ.get('SLURM_JOB_ID'),
        code_hashes={f.relative_to(Path(__file__).parent).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in Path(__file__).parent.rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')},
        official_holdout_used=False,training_performed=False,review_actor='SCRIPTED_TEST_REVIEWER',cases=[])
    for case_id,narrative in CASES:
        result=w.submit(dict(case_id=case_id,narrative=narrative),True,service)
        action=('MANUAL_REVIEW' if result['queue'].startswith('manual') else
                'PREPARE_SPECIALIST_REVIEW' if result['specialist_attention_required'] else 'PREPARE_BILLING_REVIEW')
        decision='reject' if action=='MANUAL_REVIEW' else 'accept'
        record=w.review(case_id,reviewer_id='SCRIPTED_TEST_REVIEWER',decision=decision,action_id=action,
                        reason='Synthetic interface and persistence test; not actual employee validation',
                        original_checked=True,semantic_support_checked=True)
        evidence['cases'].append(dict(case_id=case_id,narrative=narrative,result=result,review=record,
                                     persisted=w.show(case_id)['state']))
    for case_id,status,raw,narrative in [
        ('timeout','timeout',None,'An unauthorized purchase appears on my card.'),
        ('malformed','ok','{"bad":true}','My bill includes an unexpected fee.')]:
        r=w.submit(dict(case_id=case_id,narrative=narrative),True,lambda n:(status,raw,{'injected':True}))
        evidence.setdefault('failures',[]).append(dict(case_id=case_id,result=r))
    with w.connect() as c:
        evidence['audit_events']=[dict(zip(('id','case_id','at','kind','body'),row)) for row in c.execute('SELECT * FROM events')]
    evidence['authority_invariants_passed']=all(c['result']['employee_confirmation_required'] and
        not c['result']['operational_execution'] and not c['review']['operational_execution'] for c in evidence['cases'])
    evidence['live_model_valid_count']=sum(c['result']['model_output'] is not None for c in evidence['cases'])
    evidence['live_service_tested']=all(c['result']['service'].get('model')==args.model for c in evidence['cases'])
    (out/'end_to_end.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({k:evidence[k] for k in ('hostname','slurm_job','authority_invariants_passed','live_model_valid_count','live_service_tested')}),flush=True)
    for c in evidence['cases']: print(c['case_id'],c['result']['queue'],c['persisted'],flush=True)
    if not evidence['live_service_tested'] or not evidence['authority_invariants_passed'] or evidence['live_model_valid_count']!=3:
        raise SystemExit(1)

if __name__=='__main__':main()
