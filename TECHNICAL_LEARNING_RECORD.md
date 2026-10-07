# Technical Learning and Candidate Contribution Record

Repository: pingan1224/abi-complaint-management-poc
Branch: louyinuo/hw4-candidate
Candidate code commit: 785fa93c4047e9fa96cd32fb822c9b54ebaf4b7e

## Candidate contribution

I added complaint_output_validator.py, a small Python parser using the
standard-library json module. It checks that a model response is valid
JSON and contains summary, issue, urgency, routing, escalation,
human_review, and next_action fields with basic expected types. This
schema is a proposal for team review, not an approved taxonomy.

## Environment and execution evidence

Local environment: macOS, Python 3.13.6.
Model interface: Open WebUI at 172.22.42.174:8080.
Displayed model label: llama-3.1-8b-instruct.
The interface did not show quantization or inference settings. I did not
verify that the WebUI host is a DGX Spark or run the parser code there.
The model responses were generated in the WebUI and checked by the
parser locally.

| Synthetic complaint | Model output highlights | Parser result |
| --- | --- | --- |
| Unrecognized debit-card purchase | High urgency; escalation and human review true | Accepted |
| Possible duplicate mortgage payment | High urgency; escalation and human review true | Accepted |
| Monthly account fee question | Low urgency; escalation false; human review true | Accepted |

These checks establish output structure only. They do not establish
that the model's classification, urgency, routing, or recommendation is
correct.

## Failure and diagnosis

When I pasted the first model response into a file in nano, long values
were split by hard line breaks inside quoted JSON strings. The parser
rejected the file with JSONDecodeError at line 3, column 76. Inspecting
the file with line numbers showed the breaks inside issue and
next_action. Re-pasting with nano's no-wrap option produced valid JSON,
which the parser accepted. This was a copy/paste problem, not evidence
that the model returned invalid JSON.

## Verification and judgment

Python's official documentation says json.loads raises JSONDecodeError
when its input is not valid JSON and provides the error location. I
checked this against a controlled invalid-input run and valid outputs:
https://docs.python.org/3.13/library/json.html

This is a small candidate the team could consider after agreeing on an
output schema. Its main limitation is that it checks structure, not
meaning or correctness. It does not validate approved categories,
policy compliance, or whether a recommendation is appropriate.
DGX execution, direct model API integration, model build details, and
semantic evaluation remain untested.

AI helped explain Git and Python, draft the parser, and diagnose the
malformed test file. I checked the JSON behavior against Python's
documentation and controlled valid and invalid inputs.

## Self-reflection

I learned that I can make technical work manageable by choosing one
small behavior and checking it directly. AI helped me understand Git
commands and draft a Python parser, but I still needed to run it and
inspect the result. When parsing failed, I used the error location and
line-numbered file to find that I had introduced line breaks while
copying the output. Checking Python's documentation confirmed what the
JSON decoder should do. I also learned that a response passing a format
check is not proof that its banking recommendation is correct. Next
time I would record the environment and save outputs more carefully.
