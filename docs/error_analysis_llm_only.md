## Error Analysis — LLM-only (qwen2.5:7b-instruct)

Evaluated on N=1486 records. Total errors: 214 (14.4%)

### Confusion Matrix

```
  True \ Pred |     junk |  quality | quantity
----------------------------------------------
         junk |        3 |        0 |        0
      quality |       56 |     1076 |       43
     quantity |        0 |      115 |      193
```

### Error Type Breakdown

| Error Type | Count | % of Errors |
|---|---|---|
| missing_uri | 134 | 62.6% |
| ambiguous_tags | 56 | 26.2% |
| missing_metadata | 24 | 11.2% |

### Representative Misclassified Examples

### Root Cause Summary

| Cause | Mechanism | Affected Records |
|---|---|---|
| Ambiguous tags | tag1/tag2 are multi-domain service names with dual quality/metric semantics (e.g. 'score', 'rating') — LLM needs scale context to disambiguate, model sees both interpretations as valid | 56 |
| Missing agent metadata | description + OASF fields empty — model cannot infer domain, falls back to tag semantics alone | 24 |
| Missing feedbackURI | offchain_note empty — no narrative context, tag pair alone is insufficient for quality vs quantity boundary | 134 |
| Uncovered rule | Tag pattern is novel (not in rule engine vocabulary) but semantically unambiguous — rule added post-hoc would catch these | 0 |
