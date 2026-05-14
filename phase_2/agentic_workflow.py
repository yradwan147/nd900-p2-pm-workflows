"""Phase 2 — orchestrated multi-agent workflow for the Email-Router product.

We instantiate three knowledge agents (Product Manager, Program Manager,
Development Engineer) and pair each with an EvaluationAgent that re-asks
the worker until the response matches the role's structured-output
template. A RoutingAgent picks the right knowledge-agent pair for each
step of the plan, and the ActionPlanningAgent translates the high-level
workflow prompt ("What would the development tasks for this product be?")
into a stepwise plan over the three role personas.
"""

import os

from dotenv import load_dotenv

from workflow_agents.base_agents import (
    ActionPlanningAgent,
    EvaluationAgent,
    KnowledgeAugmentedPromptAgent,
    RoutingAgent,
)

# ----------------------------------------------------------------------------
# 1. Setup — API key + product spec
# ----------------------------------------------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
assert openai_api_key, "Set OPENAI_API_KEY in the environment first."

with open("Product-Spec-Email-Router.txt", "r", encoding="utf-8") as f:
    product_spec = f.read()


# ----------------------------------------------------------------------------
# 2. Action-planning agent — turns the high-level prompt into a list of steps
# ----------------------------------------------------------------------------

knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning,
)


# ----------------------------------------------------------------------------
# 3. Three worker agents + matching EvaluationAgents
# ----------------------------------------------------------------------------

# ---- Product Manager (writes user stories) --------------------------------

persona_product_manager = (
    "You are a Product Manager, you are responsible for defining the "
    "user stories for a product."
)
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, "
    "and a desired outcome. The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas "
    "are the different users of the product. \n\n"
    f"--- PRODUCT SPEC ---\n{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager,
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona="You are an evaluation agent that checks the answers of other worker agents",
    evaluation_criteria=(
        "The answer must be a list of user stories. Each story must follow "
        "the exact structure: 'As a [type of user], I want [an action or feature] "
        "so that [benefit/value].'"
    ),
    worker_agent=product_manager_knowledge_agent,
    max_interactions=5,
)


# ---- Program Manager (rolls stories into features) ------------------------

persona_program_manager = (
    "You are a Program Manager, you are responsible for defining the features "
    "for a product."
)
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into "
    "cohesive groups."
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager,
)

persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    worker_agent=program_manager_knowledge_agent,
    max_interactions=5,
)


# ---- Development Engineer (turns features into eng tasks) -----------------

persona_dev_engineer = (
    "You are a Development Engineer, you are responsible for defining the "
    "development tasks for a product."
)
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to "
    "implement each user story."
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer,
)

persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work\n"
        "Related User Story: Reference to the parent user story\n"
        "Description: Detailed explanation of the technical work required\n"
        "Acceptance Criteria: Specific requirements that must be met for completion\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first"
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=5,
)


# ----------------------------------------------------------------------------
# 4. Per-role support functions — used as RoutingAgent's `func` values
# ----------------------------------------------------------------------------


def product_manager_support_function(query: str) -> str:
    """Worker → Evaluator loop for Product Manager (user stories)."""
    result = product_manager_evaluation_agent.evaluate(query)
    return result["final_response"]


def program_manager_support_function(query: str) -> str:
    """Worker → Evaluator loop for Program Manager (features)."""
    result = program_manager_evaluation_agent.evaluate(query)
    return result["final_response"]


def development_engineer_support_function(query: str) -> str:
    """Worker → Evaluator loop for Development Engineer (tasks)."""
    result = development_engineer_evaluation_agent.evaluate(query)
    return result["final_response"]


# ----------------------------------------------------------------------------
# 5. RoutingAgent — picks the right role for each plan step
# ----------------------------------------------------------------------------

routing_agent = RoutingAgent(openai_api_key=openai_api_key, agents=[])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Define user stories from the product spec. Use this agent when a step "
            "mentions users, personas, user stories, or how a user will use the product."
        ),
        "func": product_manager_support_function,
    },
    {
        "name": "Program Manager",
        "description": (
            "Define product features by grouping user stories. Use this agent when a "
            "step mentions features, capabilities, or rolling up user stories."
        ),
        "func": program_manager_support_function,
    },
    {
        "name": "Development Engineer",
        "description": (
            "Define engineering tasks needed to build each feature or user story. "
            "Use this agent when a step mentions implementation, engineering tasks, "
            "development work, or how to build the product."
        ),
        "func": development_engineer_support_function,
    },
]


# ----------------------------------------------------------------------------
# 6. Run the workflow
# ----------------------------------------------------------------------------

print("\n*** Workflow execution started ***\n")

workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Action planner produced {len(workflow_steps)} step(s):")
for i, step in enumerate(workflow_steps, 1):
    print(f"  {i}. {step}")

completed_steps: list[str] = []
for i, step in enumerate(workflow_steps, 1):
    print(f"\n===== Step {i}/{len(workflow_steps)}: {step!r} =====")
    result = routing_agent.route(step)
    completed_steps.append(result)
    preview = result[:200].replace("\n", " ") if result else ""
    print(f"--- Step {i} result preview ---\n{preview}{'...' if len(result) > 200 else ''}")

print("\n\n========== FINAL WORKFLOW OUTPUT ==========")
print(completed_steps[-1] if completed_steps else "(no steps completed)")
