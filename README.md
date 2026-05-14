# Project 2 — Agentic Workflows for Project Management (nd900-p2)

Final deliverable for Udacity Agentic AI Nanodegree (nd900),
Project 2.

A reusable **7-class agent toolkit** + a **multi-agent orchestrated
workflow** that turns a product specification (`Email Router`) into a
fully-structured engineering plan with user stories, features, and
development tasks.

## What's in here

```
phase_1/
  workflow_agents/
    base_agents.py      ← 7 agent classes (1 pre-built + 6 implemented this PR)
    __init__.py
  direct_prompt_agent.py            ← test #1
  augmented_prompt_agent.py         ← test #2
  knowledge_augmented_prompt_agent.py ← test #3
  evaluation_agent.py               ← test #4
  routing_agent.py                  ← test #5 (Texas / Europe / Math demo)
  action_planning_agent.py          ← test #6 (egg-recipes demo)
  rag_knowledge_prompt_agent.py     ← test #7 (Clara-the-marine-biologist text)
  README.md
phase_2/
  agentic_workflow.py               ← orchestrated multi-agent run
  Product-Spec-Email-Router.txt     ← input product spec
  workflow_run.log                  ← committed log of the live 11-step run
  workflow_agents/                  ← copy of base_agents so phase_2 imports cleanly
README.md / LICENSE / .gitignore / .env.example / requirements.txt
starter_README.md                   ← original Udacity README for reference
```

## The seven agents

| Class | Pattern | Reasoning surface |
|---|---|---|
| `DirectPromptAgent` | bare LLM call | none |
| `AugmentedPromptAgent` | persona-shaped system message | LLM (parametric) |
| `KnowledgeAugmentedPromptAgent` | persona + injected closed-book knowledge | LLM (closed-book) |
| `RAGKnowledgePromptAgent` | open-book retrieval via cosine over text-embedding-3-large chunks | LLM + retrieval |
| `EvaluationAgent` | judge / refine loop wrapping any worker agent | LLM (self-critique) |
| `RoutingAgent` | route by cosine similarity to expert agent | LLM (embedding) |
| `ActionPlanningAgent` | natural-language → list of typed steps | LLM (structured-list) |

Models (rubric-mandated):
* `gpt-3.5-turbo` for every chat call
* `text-embedding-3-large` for every embedding call

## Phase 2 — orchestrated workflow

`phase_2/agentic_workflow.py` wires three knowledge agents (Product
Manager, Program Manager, Development Engineer), pairs each with a
matching EvaluationAgent (worker-judge loop), routes between them
via a RoutingAgent on cosine similarity over their descriptions, and
plans the overall flow with an ActionPlanningAgent.

The live 11-step run is committed verbatim as
[`phase_2/workflow_run.log`](phase_2/workflow_run.log) — 833 lines
of step-by-step output covering action planning → routing → worker
response → evaluator critique → refinement → final approved task.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...     # see .env.example
```

## Running

```bash
# Standalone test scripts (Phase 1)
cd phase_1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py

# Orchestrated workflow (Phase 2)
cd ../phase_2
python agentic_workflow.py
```

The seven test scripts and the orchestrated workflow each cost
under $0.05 in OpenAI usage at gpt-3.5-turbo + text-embedding-3-large
pricing.

## License

MIT.

## Test-script outputs (rubric Agent Testing)

Per the rubric's Agent Testing line ("Outputs ... are provided for the
execution of all seven test scripts"), this submission ships verbatim
terminal output for every Phase 1 test script in
[`phase_1/outputs/`](phase_1/outputs/):

```
phase_1/outputs/
  direct_prompt_agent.txt              answer + 'Knowledge source' explanation
  augmented_prompt_agent.txt           answer + persona-effect commentary
  knowledge_augmented_prompt_agent.txt answer confirms use of injected knowledge
  evaluation_agent.txt                 full 2-iteration worker/judge trace
  action_planning_agent.txt            extracted scrambled-eggs steps
  routing_agent.txt                    Texas / Europe / Math routing decisions
  rag_knowledge_prompt_agent.txt       Clara-podcast retrieval + answer
```

Each `.txt` file was produced by `python <script>.py > outputs/<script>.txt 2>&1`
from a clean working directory with `OPENAI_API_KEY` set. The outputs
collectively prove every implemented agent works end-to-end.
