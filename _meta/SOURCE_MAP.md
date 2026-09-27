# Source Map

24 video transcripts (all romanized to Latin script in `D:\AgenticAI\Evals\sources\`)
plus 7 code repositories. This file maps every source to the case study that covers it.

> Transcripts marked *(Hinglish)* were auto-transcribed in Devanagari and *(Banglish)* in
> Bengali script; both were programmatically romanized (`work/romanize.py`). Original files
> are preserved under `extracted/`. Technical English terms were restored from a curated
> phonetic dictionary, so occasional residual noise (`sistama` = system) is expected —
> context always disambiguates.

## Track A — LLM model evals, methods, benchmarks

| CS | Source file (`D:\AgenticAI\Evals\sources\`) | Domain |
|----|---------------------------------------------|--------|
| CS-01 | `Introduction_to_LLM_Evaluations_Model_Evals_vs_Application_Evals_CampusX.txt` (Hinglish) | 01-foundations |
| CS-02 | `Master_LLM_Evaluations_The_Step-by-Step_Playlist_for_2026_New_Playlist_CampusX.txt` (Hinglish) | 01-foundations |
| CS-03 | `Why_Your_AI_Application_Needs_Multiple_Eval_Pipelines_CampusX.txt` (Hinglish) | 01-foundations |
| CS-04 | `How_to_Evaluate_LLM_Applications_The_Complete_Workflow_CampusX.txt` (Hinglish) | 01-foundations |
| CS-05 | `LLM_Model_Evals_Capabilities_CampusX.txt` (Hinglish) | 01-foundations |
| CS-06 | `Offline_Evals_Vs_Online_Evals_CampusX.txt` (Hinglish) | 02-methods |
| CS-07 | `LLM_Eval_Methods_LLM-as-a-Judge_Reference_Based_Evals_Vs_Reference_Free_Evals_Ca.txt` (Hinglish) | 02-methods |
| CS-08 | `Mastering_G-Eval_The_Deterministic_LLM-as-a-Judge_Framework_Explained_CampusX.txt` (English) | 02-methods |
| CS-09 | `How_to_Use_LLM_Leaderboards_CampusX.txt` (Hinglish) | 03-benchmarks |
| CS-10 | `What_are_LLM_Benchmarks_The_Evolution_of_AI_Knowledge_Benchmarks_CampusX.txt` (Hinglish) | 03-benchmarks |
| CS-11 | `Whats_is_LLM_Benchmarking_Benchmark_Saturation_vs._Contamination_CampusX.txt` (Hinglish) | 03-benchmarks |
| CS-12 | `Selecting_the_Right_LLM_for_Your_AI_App_Running_Custom_Model_Evals_CampusX.txt` (Banglish) | 03-benchmarks |

## Track B — RAG evals, agentic evals, production evals

| CS | Source file (`D:\AgenticAI\Evals\sources\`) | Domain |
|----|---------------------------------------------|--------|
| CS-13 | `How_to_Test_RAG_RetrieversHands-On_CampusX.txt` (English) | 04-rag |
| CS-14 | `How_to_Answer_How_Do_You_Evaluate_Your_RAG_App_in_GenAI_Interviews_CampusX.txt` (Banglish) | 04-rag |
| CS-15 | `RAG_Operational_Evals_Building_Faster_Cheaper_RAG_Systems_CampusX.txt` (English) | 04-rag |
| CS-16 | `Securing_Your_RAG_Application_Testing_for_Toxicity_Leakage_Scope_Drift_CampusX.txt` (Banglish) | 04-rag |
| CS-17 | `Agentic_Evaluations_Workshop_-_Deep_Dive_on_the_Future_on_Evals_for_Agents.txt` (English) | 05-agentic |
| CS-18 | `RL_for_Agents_Workshop_-_Deep_Dive_on_Training_Agents_with_RL_and_Open_Source.txt` (English) | 05-agentic |
| CS-19 | `Training_Agents_Live_tutorial_on_how_to_fine-tune_a_coding_agent_for_continual_l.txt` (English) | 05-agentic |
| CS-20 | `Building_Evaluations_for_AI_Agents_That_Thrive_in_Prod.txt` (English) | 06-production |
| CS-21 | `Building_AI_Agents_with_Observability_Traces_Evals_Alerts_Red_Teaming_Explained.txt` (English) | 06-production |
| CS-22 | `How_to_set_Evaluation_for_AI_Agents_Scale_them.txt` (English) | 06-production |
| CS-23 | `How_to_Price_Your_AI_Agents_The_Framework_Companies_Use_Sierra_Decagon_Finn.txt` (English) | 06-production |
| CS-24 | `The_n8n_Limitation_Claude_Solves_How_Claude_Code_Accelerates_Everything.txt` (English) | 06-production |

## Code repositories (`D:\AgenticAI\Evals\extracted\`)

| Dossier | Repo | Stack | Scale | Role |
|---------|------|-------|-------|------|
| CODE-01 | `awesome-evals-main/awesome-evals-main` | Markdown | 149 md | Curated canon + `PATTERNS.md` playbook of runnable eval patterns |
| CODE-02 | `evals-main/evals-main` | Python | 353 py | OpenAI Evals — the registry/YAML eval framework |
| CODE-03 | `evals-skills-main/evals-skills-main` | Markdown | 11 md | Agent-skill packaging of eval workflows (Claude/Codex plugins) |
| CODE-04 | `evalscope-main/evalscope-main` | Python + TS | 1363 py | ModelScope Evalscope — benchmark runner (perf, VLM, arena, agent) |
| CODE-05 | `frontier-evals-main/frontier-evals-main` | Python + md | 417 py, 660 md | Frontier-lab style problem-set evals for research-grade tasks |
| CODE-06 | `langfuse-main/langfuse-main` | TypeScript | 5190 ts | LLM observability + eval platform (traces, scores, datasets, experiments) |
| CODE-07 | `search_evals-main/search_evals-main` | Python | 27 py | Search/RAG eval harness (retrieval + answer quality) |

> `evalscope-main (1)` is a byte-identical duplicate of `evalscope-main` and is ignored.
