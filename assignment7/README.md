# Assignment 7 candidate human authority gate

This standard-library component is a small proposed integration boundary for the employee-facing complaint PoC. It consumes the six-field suggestion schema in the Team 5 **candidate** model runner. The team's live intake gate, interface, workflow, and policy still require workshop approval.

`apply_authority_gate(scope_verified=..., model_suggestion=...)` returns `queue`, `suggested_route`, `reasons`, and `final_action_authorized`. The final field is always `false`. The caller must obtain `scope_verified` from an independent intake check, then record the employee's final confirm, override, reject, or defer action outside this component. Never infer scope from a model response or a retrospective course label.

The function withholds a model route if scope is unverified, the service fails, or the output is malformed. A fraud/security route, model-suggested escalation, model-suggested specialist review, or stated uncertainty moves the case to `specialist_review`; a routine valid suggestion goes to `employee_review`. The gate does not examine the original narrative, make a fraud finding, detect a risk the model failed to mention, or prove that a specialist queue exists. Human confirmation remains mandatory for all cases.

Run local boundary tests from this directory with Python 3.10 or newer:

```bash
python -B -m unittest test_authority_gate.py -v
```

This component makes **no model-service call**. On 28 October 2026, the fixed candidate commit passed four tests in DGX JupyterLab, and a synthetic six-field result accepted by the fixed Team 5 parser passed into this gate; see [DGX_TEST_EVIDENCE.md](DGX_TEST_EVIDENCE.md). A live DGX model-service request and an end-to-end employee workflow test remain unperformed. The team must reconcile its approved schema and queue policy before adopting the gate.
