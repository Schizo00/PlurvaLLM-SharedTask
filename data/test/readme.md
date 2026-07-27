# PlurVA-LLM Shared Task Datasets

This folder contains the test data of Chinese, Indonesian, and Sri Lankan value datasets for the shared task. All files are in JSONL format。

## Test Data Format

Column names:

- `ID`: Question ID with a language prefix.
  - Chinese: `ZH_`
  - Indonesian: `ID_`
  - Sri Lankan: `SI_`
- `Value`: Original value label or value taxonomy.
- `Value_English`: English value name or English value taxonomy.
- `Scenario`: Scenario text. This field is only populated for Indonesian data; it is `""` for Chinese and Sri Lankan data.
- `Question`: Question text.
- `Option_A`: Answer option A.
- `Option_B`: Answer option B.
- `Option_C`: Answer option C. This field is `""` for Sri Lankan data.
- `Option_D`: Answer option D. This field is `""` for Sri Lankan data.


## Evaluation Procedure

1. For the Chinese and Indonesian datasets, the output should be one of: “A”, “B”, “C”, or “D”.
2. For the Sri Lankan dataset, the output should be one of: “A”, “B”, “Both”, or “0”, where:
  a. “Both” indicates that both options are correct.
  b. “0” indicates that neither option is acceptable given the question.


## Files and Counts

| Language | Test |
|---|---:|
| Chinese | 3210 |
| Indonesian | 1468 |
| Sri Lankan | 797 |


