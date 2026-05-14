"""Test script for AugmentedPromptAgent (persona-shaped system message)."""
import os

from dotenv import load_dotenv

from workflow_agents.base_agents import AugmentedPromptAgent

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_agent = AugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona)
augmented_agent_response = augmented_agent.respond(prompt)

print(augmented_agent_response)

# Knowledge-source / persona explanation (rubric TODO #4)
print(
    "\n[Knowledge source] As with DirectPromptAgent the answer 'Paris' "
    "comes from the LLM's parametric knowledge — there is no injected "
    "knowledge here. However the system prompt now sets a *persona* "
    "('college professor, answer starts with Dear students,') which "
    "demonstrably shapes the *form* of the answer: the LLM opens with "
    "'Dear students,' as instructed even though the underlying fact "
    "(Paris is the capital of France) is unchanged."
)
