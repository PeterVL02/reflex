"System-one prompt classification with Laya (convaiinnovations/laya), run locally."

import json
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from laya import Router
from laya.agent import Agent
from laya.agent import load as load_agent

# Laya gives the question ~192 tokens (English checkpoint), so instructions must stay short.
PROMPTS_PATH: Path = Path(__file__).with_name("sys_one_prompt.json")
ENGLISH_MODEL: str = "convaiinnovations/laya"
HIDDEN_PROMPT: str = "v2_attack_min"  # the best zero-shot instruction (AUROC 0.92)


@cache
def load_router() -> Router:
    """Load the Laya router once (checkpoints download on first use, GPU if available)."""
    return Router()


def load_instructions(prompt_name: str) -> str:
    """
    Read one Laya instruction string from sys_one_prompt.json.

    Args:
        prompt_name (str): Key of the instruction in the JSON file.

    Returns:
        str: The instruction text.

    Raises:
        KeyError: If `prompt_name` is not in the JSON file.
    """
    prompts: dict[str, str] = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    return prompts[prompt_name]


def classify_prompts(
    prompts: list[str] | pd.Series[str],
    prompt_name: str = "v1_attack_harm",
    batch_size: int = 32,
) -> npt.NDArray[np.float64]:
    """
    Score prompts with Laya's yes/no ("noul") question.

    Args:
        prompts (list[str] | pd.Series[str]): Texts to classify.
        prompt_name (str): Key of the instruction in sys_one_prompt.json.
        batch_size (int): Prompts per forward pass.

    Returns:
        npt.NDArray[np.float64]: Probability that each prompt is malicious (label 1).
    """
    questions = {
        "malicious": {"type": "noul", "instructions": load_instructions(prompt_name)}
    }
    requests = [{"state": p, "questions": questions} for p in prompts]
    results = load_router().predict_batch(requests, batch_size=batch_size)
    return np.array([r["answers"]["malicious"]["noul"] for r in results])


@cache
def load_english_agent() -> Agent:
    """Load the English Laya checkpoint once, so every hidden state shares its 1024-d space."""
    return load_agent(ENGLISH_MODEL)


def extract_hidden_states(
    prompts: list[str] | pd.Series[str], prompt_name: str = HIDDEN_PROMPT
) -> npt.NDArray[np.float32]:
    """
    Return Laya's decision-head hidden states at the two noul answer markers per prompt.

    These are the vectors Laya's scorer turns into the "false"/"true" logits: the encoder
    output after the 2-layer decision head, read at each option's [MASK] position. They're
    conditioned on the instruction, unlike a generic text embedding. One prompt per forward
    pass, so each hooked row belongs to exactly one prompt.

    Args:
        prompts (list[str] | pd.Series[str]): Texts to encode.
        prompt_name (str): Key of the instruction in sys_one_prompt.json.

    Returns:
        npt.NDArray[np.float32]: Shape (len(prompts), 2 * 1024): the "false" marker's state,
        then the "true" marker's.
    """
    agent = load_english_agent()
    questions = {
        "malicious": {"type": "noul", "instructions": load_instructions(prompt_name)}
    }
    captured: list[torch.Tensor] = []
    hook = agent.model.scorer.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs[0].detach().float().cpu())
    )
    try:
        for prompt in prompts:
            agent.predict_batch([prompt], questions, batch_size=1)
    finally:
        hook.remove()
    if len(captured) != len(prompts) or {c.shape[:2] for c in captured} != {(1, 2)}:
        raise RuntimeError("hooked hidden states don't align one-to-one with prompts")
    return torch.cat([c.reshape(1, -1) for c in captured]).numpy()
