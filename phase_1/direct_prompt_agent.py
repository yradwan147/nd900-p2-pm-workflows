"""Test script for DirectPromptAgent (no system message, no persona).

This demonstrates the agent answering an unaided question using only
the LLM's parametric knowledge.
"""
import os

from dotenv import load_dotenv

from workflow_agents.base_agents import DirectPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

prompt = "What is the Capital of France?"

direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

print(direct_agent_response)

# Knowledge-source explanation (rubric TODO #5)
print(
    "\n[Knowledge source] The DirectPromptAgent issues a bare user-message "
    "request to gpt-3.5-turbo with NO system prompt and NO injected "
    "knowledge. The answer therefore comes from the LLM's pretraining "
    "data (parametric knowledge) — there is no retrieval, no persona, "
    "no custom context."
)
