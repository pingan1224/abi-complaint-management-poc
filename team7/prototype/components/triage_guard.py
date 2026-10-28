"""Offline candidate authority guard. No model calls or operational actions."""
ROUTES = frozenset({'Card Billing Disputes', 'Card Fraud & Security'})
URGENCIES = frozenset({'Low', 'Medium', 'High', 'Critical'})
STATUSES = frozenset({'ok', 'timeout', 'unavailable'})
KEYS = frozenset({'summary', 'suggested_route', 'urgency', 'escalation', 'abstain', 'rationale'})

def triage_guard(*, admitted, model_status, model_output, safety_flags=()):
    """Return a review disposition; all model values are untrusted suggestions.

    admitted is a server-side bool, never an LLM or historical-label scope test.
    safety_flags is a tuple of reason codes from an independent versioned check.
    Unknown/broken inputs raise ValueError; caller must retain intake for review.
    """
    if type(admitted) is not bool or type(model_status) is not str or model_status not in STATUSES:
        raise ValueError('invalid admission or service status; retain for manual review')
    if not isinstance(safety_flags, tuple) or any(type(f) is not str or not f.strip() for f in safety_flags):
        raise ValueError('invalid safety flags; retain for manual review')
    out = dict(disposition='manual_triage', suggestion=None, employee_confirmation_required=True,
               specialist_review_required=False, execute_action=False, reasons=[])
    if not admitted:
        out.update(disposition='out_of_scope_review', reasons=['not_admitted'])
    elif model_status != 'ok':
        out['reasons'] = ['model_' + model_status]
    elif (type(model_output) is not dict or set(model_output) != KEYS
          or any(type(model_output[k]) is not str or not model_output[k].strip()
                 for k in ('summary', 'rationale'))
          or type(model_output['suggested_route']) is not str
          or model_output['suggested_route'] not in ROUTES
          or type(model_output['urgency']) is not str
          or model_output['urgency'] not in URGENCIES
          or type(model_output['escalation']) is not bool
          or type(model_output['abstain']) is not bool):
        out['reasons'] = ['invalid_model_output']
    elif model_output['abstain']:
        out['reasons'] = ['model_abstained']
    else:
        out.update(disposition='employee_review', suggestion=dict(model_output))
        if model_output['escalation'] or model_output['urgency'] in {'High', 'Critical'} or model_output['suggested_route'] == 'Card Fraud & Security':
            out.update(disposition='specialist_review', specialist_review_required=True,
                       reasons=['model_risk_signal'])
    if safety_flags:
        out['specialist_review_required'] = True
        out['reasons'] += ['independent_flag:' + f for f in safety_flags]
        if admitted:
            out['disposition'] = 'specialist_review'
    return out
