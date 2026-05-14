"""Reusable agent toolkit for Udacity Agentic AI nd900 Project 2.

Seven agent classes implementing the patterns covered in the
Agentic Workflows course:

  * DirectPromptAgent          — bare LLM call, no system message
  * AugmentedPromptAgent       — persona via a system message
  * KnowledgeAugmentedPromptAgent — persona + closed-book knowledge
  * RAGKnowledgePromptAgent    — open-book retrieval over a corpus
  * EvaluationAgent            — judge/refine loop around a worker
  * RoutingAgent               — cosine-similarity router over experts
  * ActionPlanningAgent        — natural-language → list[step]

All agents talk to the public OpenAI API (`api.openai.com`) using
the `OPENAI_API_KEY` the caller passes in. The rubric mandates
`gpt-3.5-turbo` for chat and `text-embedding-3-large` for
embeddings, which we honour throughout.
"""
from __future__ import annotations

import csv
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI


# ---------------------------------------------------------------------------
# DirectPromptAgent — bare LLM call, no system message
# ---------------------------------------------------------------------------


class DirectPromptAgent:
    """Pass the user prompt straight to gpt-3.5-turbo with no system role."""

    def __init__(self, openai_api_key: str) -> None:
        self.openai_api_key = openai_api_key

    def respond(self, prompt: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# AugmentedPromptAgent — system-message persona, then user prompt
# ---------------------------------------------------------------------------


class AugmentedPromptAgent:
    """Wrap a DirectPromptAgent with a persona-shaped system message."""

    def __init__(self, openai_api_key: str, persona: str) -> None:
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona}. Forget any previous context and "
            f"respond from the perspective of this persona only."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# KnowledgeAugmentedPromptAgent — closed-book persona + injected facts
# ---------------------------------------------------------------------------


class KnowledgeAugmentedPromptAgent:
    """Persona-shaped agent that must answer ONLY from injected knowledge."""

    def __init__(self, openai_api_key: str, persona: str, knowledge: str) -> None:
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text: str) -> str:
        client = OpenAI(api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: "
            f"{self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# RAGKnowledgePromptAgent — open-book retrieval over a chunked corpus
# (Pre-built by Udacity; we keep the implementation but swap the
# Vocareum base_url for the public OpenAI endpoint so the agent
# works against the user-provided OPENAI_API_KEY.)
# ---------------------------------------------------------------------------


class RAGKnowledgePromptAgent:
    """Retrieval-augmented agent over a chunked text corpus.

    Uses `text-embedding-3-large` (rubric-mandated) for both the
    corpus chunks and the incoming query, picks the highest-cosine-
    similarity chunk, and answers via `gpt-3.5-turbo` constrained to
    that single chunk.
    """

    def __init__(self, openai_api_key: str, persona: str,
                 chunk_size: int = 2000, chunk_overlap: int = 100) -> None:
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text: str) -> list[float]:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two) -> float:
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def chunk_text(self, text: str):
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
        chunks, start, chunk_id = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)
            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end,
            })
            if end == len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1
        with open(f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt: str) -> str:
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )
        best_chunk = df.loc[df["similarity"].idxmax(), "text"]
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# EvaluationAgent — judge a worker's response, ask for corrections, loop
# ---------------------------------------------------------------------------


class EvaluationAgent:
    """Iteratively refine a worker agent's response.

    Wraps a worker agent (any object with a `respond(prompt)` method)
    and a free-text evaluation criterion. On each iteration the
    evaluator (a) asks the worker for a response, (b) judges it
    Yes/No, and (c) if No, asks an LLM to draft correction
    instructions and feeds those back to the worker.
    """

    def __init__(self, openai_api_key: str, persona: str,
                 evaluation_criteria: str, worker_agent,
                 max_interactions: int = 10) -> None:
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt: str) -> dict:
        client = OpenAI(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        evaluation = ""
        response_from_worker = ""
        i = 0  # Pre-bind so the post-loop return doesn't reference an unbound name.

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i + 1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}."},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}."},
                    {"role": "user", "content": instruction_prompt},
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                f"It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1,
        }


# ---------------------------------------------------------------------------
# RoutingAgent — cosine-similarity router over a list of (name, description, func)
# ---------------------------------------------------------------------------


class RoutingAgent:
    """Route a user input to one of N expert agents by embedding similarity.

    `agents` is a list of dicts with `name`, `description`, and `func`
    keys. `func` must take a single string argument and return the
    agent's response.
    """

    def __init__(self, openai_api_key: str, agents: list[dict]) -> None:
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text: str) -> list[float]:
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def route(self, user_input: str) -> str:
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue
            similarity = float(
                np.dot(input_emb, agent_emb)
                / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            )
            print(similarity)
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ---------------------------------------------------------------------------
# ActionPlanningAgent — natural language → list[step]
# ---------------------------------------------------------------------------


class ActionPlanningAgent:
    """Extract a clean step-by-step plan from a free-text request."""

    def __init__(self, openai_api_key: str, knowledge: str) -> None:
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> list[str]:
        client = OpenAI(api_key=self.openai_api_key)
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from "
            "the user prompt the steps requested to complete the action the user is "
            "asking for. You return the steps as a list. Only return the steps in your "
            "knowledge. Forget any previous context. This is your knowledge: "
            f"{self.knowledge}"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content

        # Strip empty lines, list markers, and headers.
        raw_lines = response_text.split("\n")
        steps: list[str] = []
        for line in raw_lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            # drop leading bullets / numbering ("1.", "1)", "-", "*")
            cleaned = re.sub(r"^[\-\*]\s+", "", cleaned)
            cleaned = re.sub(r"^\d+[\.\)]\s+", "", cleaned)
            # drop markdown headers
            if cleaned.startswith("#"):
                continue
            steps.append(cleaned)
        return steps
