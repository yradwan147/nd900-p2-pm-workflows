"""Test script for EvaluationAgent.

We wire an EvaluationAgent around a KnowledgeAugmentedPromptAgent
that has been given an intentionally wrong knowledge string. The
evaluation criterion is 'the answer should be solely the name of a
city, not a sentence', so the first worker response (which will be
a sentence beginning with 'Dear students,') is expected to be
rejected. After the iterative refinement loop the final response
should be a single-word city name.
"""
import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

prompt = "What is the capital of France?"

# Parameters for the worker agent
persona_worker = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_worker,
    knowledge=knowledge,
)

# Parameters for the evaluation agent
persona_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_eval,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10,
)

result = evaluation_agent.evaluate(prompt)

print("\n========== FINAL RESULT ==========")
print(f"final_response : {result['final_response']!r}")
print(f"evaluation     : {result['evaluation']!r}")
print(f"iterations     : {result['iterations']}")
