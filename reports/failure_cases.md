# Failure Analysis Postmortem (Phase 2: Economic View)
**Date:** 2026-02-05
**Dataset:** Severe Drift Scenario
**Total Leaked Fraud Cases:** 19 (out of 290 total fraud)
**Total Leakage Cost (Realized Loss):** $1,855.13
## Why did we miss them?
- **Average Model Confidence on Leaks:** 0.0157
- **Average OOD Score on Leaks:** 0.0726

## Leakage by Category
- Clothing: 13 cases
- Electronics: 4 cases
- Beauty: 1 cases
- Home & Garden: 1 cases

## Top 5 Leaked Examples (Highest Cost)
| order_id     |   order_value | product_category   |   returns_velocity_30d |   fraud_prob |   ood_score |
|:-------------|--------------:|:-------------------|-----------------------:|-------------:|------------:|
| ORD-00003077 |        232.49 | Clothing           |                      0 |    0.0145348 |  0.0841174  |
| ORD-00004707 |        223.4  | Clothing           |                      0 |    0.0101457 |  0.00938282 |
| ORD-00000650 |        168.89 | Beauty             |                      0 |    0.0115083 |  0.0181094  |
| ORD-00001245 |        168.64 | Clothing           |                      0 |    0.0187329 |  0.0971113  |
| ORD-00002822 |        145.45 | Clothing           |                      0 |    0.0160582 |  0.0949956  |

## Root Cause Hypothesis