# Wave assignment: 05_ai_platforms_data_systems_and_production_engineering

## Unit `05_ai_platforms_data_systems_and_production_engineering`
- existing cases (0): none
- existing sheets (1): 01_Distributed_Training_Checkpoint_and_Recovery.md
- promised case rows (14):
  - Size an inference cluster
  - Profile memory-bound training
  - Train a large model on limited cluster capacity
  - Recover from checkpoint failures
  - Multi-model inference gateway
  - Reduce p99 LLM latency
  - Deploy a credit model safely
  - Feature freshness incident response
  - Broken upstream schema incident
  - Silent label corruption investigation
  - RAG quality regression
  - Agent tool-use failure analysis
  - Internal AI platform design
  - Cloud-cost anomaly response
- index sections: cases 7/7, sheets 11/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/23_gpu_architecture_accelerators_and_ai_hardware`
- existing cases (2): 01_Size_an_Inference_Cluster.md, 02_Profile_Memory_Bound_Training.md
- existing sheets (2): 01_H100_Architecture_and_Resource_Limits.md, 01_Roofline_Arithmetic_Intensity_and_Kernel_Diagnosis.md
- promised case rows (4):
  - Profile memory-bound training
  - Balance interactive inference latency and admitted throughput
  - profiling and parity validation with distributed-training cases
  - batching and KV accounting with Module 25
- index sections: cases 7/7, sheets 11/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/24_distributed_training_data_pipelines_and_ml_compute`
- existing cases (0): none
- existing sheets (0): none
- promised case rows (4):
  - Train a large model on limited cluster capacity
  - Recover from checkpoint failures
  - Gate retraining on corpus provenance and long-tail quality
  - Allocate a domain-training budget with restart headroom
- promised sheet rows (2):
  - Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage
  - Scaling-law proxy sweeps; licensed corpus splits; parallelism; memory accounting; checkpoint recovery
- index sections: cases 7/7, sheets 0/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/25_model_serving_inference_optimisation_and_cost`
- existing cases (0): none
- existing sheets (0): none
- promised case rows (6):
  - Multi-model inference gateway
  - Reduce p99 LLM latency
  - Select model routes within hard customer constraints
  - Customize tenant assistants without crossing isolation boundaries
  - Balance interactive inference latency and admitted throughput
  - Separate extraction correctness from routing economics
- promised sheet rows (4):
  - Eligibility before scoring; capability adapters; health-aware routing; bounded retry budgets
  - LoRA lifecycle; tenant-scoped data; adapter cache keys; session pinning; compatible base checkpoints
  - Continuous batching; chunked prefill; paged KV storage; prefix authorization; speculative-decoding correctness
  - Schema versioning; semantic checks; dead-letter replay; idempotency; quality-constrained tiered RAG routing
- index sections: cases 7/7, sheets 0/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/26_mlops_feature_platforms_and_model_lifecycle`
- existing cases (0): none
- existing sheets (1): 01_ML_Release_and_Rollback_Checklist.md
- promised case rows (3):
  - Deploy a credit model safely
  - Feature freshness incident response
  - Release and recover an agent as a compatible bundle
- promised sheet rows (1):
  - Immutable release manifests; compatibility gates; side-effect-free shadowing; staged canaries; incident containment
- index sections: cases 7/7, sheets 0/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/27_data_quality_observability_and_reliability_engineering`
- existing cases (0): none
- existing sheets (1): 01_Data_Observability_and_Incident_Triage.md
- promised case rows (4):
  - Broken upstream schema incident
  - Silent label corruption investigation
  - Gate retraining on corpus provenance and long-tail quality
  - Rebalance caches without overwhelming the authoritative store
- promised sheet rows (2):
  - Dataset lineage; synthetic-data feedback; deduplication limits; language and domain coverage
  - Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones
- index sections: cases 7/7, sheets 0/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/28_ai_evaluation_monitoring_and_continuous_improvement`
- existing cases (0): none
- existing sheets (1): 04_Model_Evaluation_and_Production_Monitoring.md
- promised case rows (4):
  - RAG quality regression
  - Agent tool-use failure analysis
  - Gate retraining on corpus provenance and long-tail quality
  - Block assistant releases on auditable quality and safety gates
- index sections: cases 7/7, sheets 11/11

## Unit `05_ai_platforms_data_systems_and_production_engineering/29_api_platform_engineering_cloud_native_and_finops`
- existing cases (0): none
- existing sheets (1): 06_25_Common_Issues.md
- promised case rows (4):
  - Internal AI platform design
  - Cloud-cost anomaly response
  - Select model routes within hard customer constraints
  - Bound global API admission with token leases
- promised sheet rows (2):
  - reference-informed routing brief
  - reference-informed token lease brief
- index sections: cases 7/7, sheets 11/11
