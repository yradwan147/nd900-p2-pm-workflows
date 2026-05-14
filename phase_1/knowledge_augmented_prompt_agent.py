"""Test script for KnowledgeAugmentedPromptAgent.

We deliberately inject a factually incorrect 'knowledge' string
('The capital of France is London') so we can verify that the agent
prefers the injected knowledge over its own parametric knowledge —
that is the whole point of a knowledge-augmented prompt.
"""
import os

from dotenv import load_dotenv

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

prompt = "What is the capital of France?"
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge,
)
response = knowledge_agent.respond(prompt)

print(response)
print(
    "\n[Demonstration] The injected knowledge string explicitly states "
    "the capital is London (the contrary of fact). The KnowledgeAugmentedPromptAgent's "
    "system message instructs the model to use ONLY the injected knowledge — so the "
    "response above should claim London, not Paris. That is the difference between "
    "this agent and the DirectPromptAgent: closed-book on injected facts."
)
