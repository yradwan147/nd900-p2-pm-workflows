"""Test script for RoutingAgent.

We define three KnowledgeAugmentedPromptAgents — Texas, Europe, Math
— and a RoutingAgent on top. The RoutingAgent embeds each agent's
description and the incoming user prompt, picks the highest-cosine
agent, and forwards the prompt.

Three prompts cover the three branches:
  - 'Tell me about the history of Rome, Texas'   -> Texas agent
  - 'Tell me about the history of Rome, Italy'   -> Europe agent
  - 'One story takes 2 days, and there are 20 stories' -> Math agent
"""
import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

persona_prof = "a college professor"

texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_prof,
    knowledge="You know everything about Texas",
)
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_prof,
    knowledge="You know everything about Europe",
)
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="a college math professor",
    knowledge=(
        "You know everything about math, you take prompts with numbers, "
        "extract math formulas, and show the answer without explanation"
    ),
)

routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents=[])
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x),
    },
]
routing_agent.agents = agents

for query in [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]:
    print(f"\n========== Routing: {query!r} ==========")
    result = routing_agent.route(query)
    print(f"Answer: {result}")
