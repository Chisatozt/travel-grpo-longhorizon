# Travel GRPO

A scaffold for reproducible post-training and evaluation of long-horizon travel-assistant agents.

> Status: **early development**. The pinned UserBench snapshot and reproducible task partitioning are implemented. Teacher trajectory collection, SFT, GRPO, model serving, and final rollouts are not implemented yet.

## Intended pipeline

```text
teacher trajectory collection
  -> replay and quality filtering
  -> action-only LoRA SFT
  -> online GRPO in UserBench
  -> frozen-test comparison of Baseline / SFT / GRPO
```

The actor, training-time user simulator, and formal evaluation simulator are intentionally separate runtime boundaries.

Generated split records follow the five-field contract in `data/example.jsonl`:
`task_id`, `composition`, `difficulty`, `source_split`, and `prompt`. The
project-level split is represented by the artifact path.

## UserBench

`environments/UserBench/` contains a pinned snapshot of Salesforce AI Research's [UserBench](https://github.com/SalesforceAIResearch/UserBench), a Gymnasium environment for multi-turn preference elicitation, simulated travel search, and recommendation. See `environments/UserBench/EMBEDDED_SOURCE.json` for provenance and licensing details.

The repository layout is structurally inspired by [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon). No training or evaluation results are claimed at this stage.

The root project license has not been selected. The embedded UserBench copyright and Apache-2.0 license remain unchanged.
