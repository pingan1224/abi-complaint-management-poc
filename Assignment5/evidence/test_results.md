# Assignment 5 — Test Results

## Data Quality Validator

The validator was executed locally against the 8,000-record Development Dataset.

### Dataset Validation

| Check | Result |
|---|---|
| Dataset rows | 8,000 |
| Dataset columns | 29 |
| Required columns | PASS |
| Input / target boundary | PASS |
| Duplicate narrative check | FLAG |

The duplicate check identified 7 distinct duplicated complaint narratives affecting 14 rows.

### Scoped Cohort Validation

The validator filtered the Development Dataset to the proposed capability scope:

- Product: Credit card
- Issue: Billing disputes
- Submitted via: Web

The resulting scoped cohort contains 653 cases.

| Label status | Cases |
|---|---:|
| Clear | 401 |
| Judgment Required | 237 |
| Insufficient Evidence | 15 |

| Urgency | Cases |
|---|---:|
| Low | 0 |
| Medium | 381 |
| High | 171 |
| Critical | 101 |

Within the scoped cohort:

- `sub_product`: 653/653 missing
- `sub_issue`: 653/653 missing

### Boundary Unit Tests

The reusable validator was tested using Python's built-in assertion mechanism.

Results:

```text
PASS: test_input_target_boundary_passes
PASS: test_required_columns_are_detected
PASS: test_missing_required_column_is_detected
PASS: test_boundary_detects_leakage_configuration

4/4 tests passed.
