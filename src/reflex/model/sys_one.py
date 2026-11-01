"System-one prompt classification with Laya (convaiinnovations/laya), run locally."

import json
from functools import cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from laya import Router

# Laya gives the question ~192 tokens (English checkpoint), so instructions must stay short.
PROMPTS_PATH: Path = Path(__file__).with_name("sys_one_prompt.json")


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
