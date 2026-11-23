"""Turn the Large Language Monkeys samples into compact answer-label arrays.

Source: ScalingIntelligence/monkey_business on Hugging Face (MIT licence), 10,000 sampled
solutions per problem with a correctness flag graded by the dataset authors.

For every problem the output holds an int16 array of length n_samples:
    0   the sample is graded correct (all correct samples are one answer class,
        since the grader judged each of them equivalent to the reference answer)
    -1  no final answer could be extracted (the sample costs a call but casts no vote)
    k>0 a wrong answer; samples with the same normalised final answer share k

    python scripts/prepare_monkey_business.py --out data/            # all configs below
    python scripts/prepare_monkey_business.py --out data/ --only GSM8K_Llama-3-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import ijson
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "ScalingIntelligence/monkey_business"
CONFIGS = [
    "GSM8K_Llama-3-8B-Instruct",
    "GSM8K_Llama-3-70B-Instruct",
    "MATH_Llama-3-70B-Instruct",
    "MATH_Llama-3-8B-Instruct",
    "MATH_Llama-3-8B",
    "MATH_Gemma-7B",
    "MATH_Gemma-2B",
]

_GSM8K = re.compile(r"####\s*([^\n]+)")
_FINAL = re.compile(r"The final answer is\s*(.+?)(?:\.\s*I hope it is correct|\s*$)", re.S)


def _last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalise(ans: str) -> str:
    """Light string normalisation so that trivially different spellings vote together."""
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".").rstrip(",")
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def extract(task: str, sample: str) -> str | None:
    if task == "GSM8K":
        m = _GSM8K.findall(sample)
        return normalise(m[-1]) if m else None
    m = _FINAL.findall(sample)
    if m:
        return normalise(m[-1])
    b = _last_boxed(sample)
    return normalise(b) if b is not None else None


def labels_for(task: str, samples: list[str], correct: list[bool]) -> np.ndarray:
    ids: dict[str, int] = {}
    out = np.empty(len(samples), dtype=np.int16)
    for i, (s, c) in enumerate(zip(samples, correct)):
        if c:
            out[i] = 0
            continue
        a = extract(task, s)
        if a is None or a == "":
            out[i] = -1
        else:
            out[i] = ids.setdefault(a, len(ids) + 1)
    return out


def prepare(config: str, out_dir: Path) -> Path:
    task = config.split("_", 1)[0]
    local = Path(os.environ.get("MONKEY_DIR", "")) / f"{config}.json"
    path = local if local.is_file() else hf_hub_download(REPO, f"{config}.json", repo_type="dataset")
    labels, meta = [], []
    with open(path, "rb") as f:
        for rec in ijson.items(f, "item"):
            labels.append(labels_for(task, rec["samples"], rec["is_corrects"]))
            meta.append({"orig_dset_idx": int(rec["orig_dset_idx"]), "split": rec["orig_dset_split"]})
    n = min(len(x) for x in labels)
    arr = np.stack([x[:n] for x in labels])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{config}.npz"
    np.savez_compressed(dest, labels=arr)
    (out_dir / f"{config}.json").write_text(json.dumps({"source": REPO, "config": config, "problems": meta}, indent=1))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--only", nargs="*", choices=CONFIGS)
    args = ap.parse_args()
    for config in args.only or CONFIGS:
        dest = prepare(config, args.out)
        z = np.load(dest)["labels"]
        print(f"{config}: {z.shape[0]} problems x {z.shape[1]} samples, "
              f"correct {np.mean(z == 0):.3f}, no answer {np.mean(z == -1):.3f} -> {dest}")


if __name__ == "__main__":
    main()
