"""Turn the sc-genrm-scaling solution sets into answer-label arrays, with token counts.

Sources (Hugging Face, Apache-2.0), released with Singhi et al., "When To Solve, When To
Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning"
(COLM 2025):
    AIME25_Solutions_QwQ-32B                       30 problems x 512 samples
    GPQA_diamond_Solutions_Llama-3.3-70B-Instruct  64 problems x 256 samples
    MATH128_Solutions_Llama-3.3-70B-Instruct       255 problems x 256 samples
    MATH128_Solutions_Qwen-2.5-7B-Instruct         128 problems x 1,024 samples
    MATH128_Solutions_Llama-3.1-8B-Instruct        128 problems x 16,384 (first 1,024 kept)

The sets come without correctness flags, so this script grades them:
    AIME   last \\boxed{...} as an integer, compared with the integer reference
    GPQA   the letter after "The final answer is", compared with the reference letter
    MATH   math-verify on the extracted answer against the reference's last \\boxed{...}
Labels follow scripts/prepare_monkey_business.py: 0 correct, -1 no answer, k > 0 wrong
answers grouped by normalised string. For QwQ-32B the output also holds each sample's
length in QwQ tokens (Qwen/QwQ-32B tokenizer), so rules can be compared on tokens.

    python scripts/prepare_sc_genrm.py --out data
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import HfApi, hf_hub_download

ORG = "sc-genrm-scaling"
SETS = {
    "AIME25_QwQ-32B": ("AIME25_Solutions_QwQ-32B", "AIME", None),
    "GPQA_Llama-3.3-70B-Instruct": ("GPQA_diamond_Solutions_Llama-3.3-70B-Instruct", "GPQA", None),
    "MATH_Llama-3.3-70B-Instruct": ("MATH128_Solutions_Llama-3.3-70B-Instruct", "MATH", None),
    "MATH_Qwen-2.5-7B-Instruct": ("MATH128_Solutions_Qwen-2.5-7B-Instruct", "MATH", None),
    "MATH_Llama-3.1-8B-Instruct": ("MATH128_Solutions_Llama-3.1-8B-Instruct", "MATH", 1024),
}
TOKENIZER = {"AIME25_QwQ-32B": "Qwen/QwQ-32B"}

_FINAL = re.compile(r"[Tt]he final answer is[:\s]*(.+?)(?:\.\s*I hope it is correct|\.?\s*$)", re.S)
_LETTER = re.compile(r"[Tt]he final answer is[:\s]*\(?\$?\\?(?:boxed\{)?\(?([A-D])\b")


def last_boxed(text: str) -> str | None:
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
    s = ans.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("$", "")
    s = re.sub(r"\s+", "", s).rstrip(".").rstrip(",")
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def extract(task: str, sample: str) -> str | None:
    if task == "GPQA":
        m = _LETTER.findall(sample)
        return m[-1] if m else None
    if task == "AIME":
        b = last_boxed(sample)
        if b is None:
            return None
        digits = re.sub(r"[^\d]", "", normalise(b))
        return str(int(digits)) if digits else normalise(b)
    b = last_boxed(sample)
    if b is None:
        m = _FINAL.findall(sample)
        b = m[-1] if m else None
    return normalise(b) if b else None


def reference(task: str, gt: str) -> str:
    if task == "MATH":
        b = last_boxed(gt)
        return b if b is not None else gt
    return gt.strip()


class Grader:
    """Grade each distinct extracted answer once per problem."""

    def __init__(self, task: str, gt: str) -> None:
        self.task, self.ref, self.cache = task, reference(task, gt), {}
        if task == "MATH":
            from math_verify import parse
            self.parsed_ref = parse(f"${self.ref}$")

    def __call__(self, ans: str) -> bool:
        if ans not in self.cache:
            if self.task == "AIME":
                ok = ans.isdigit() and int(ans) == int(re.sub(r"[^\d]", "", self.ref))
            elif self.task == "GPQA":
                ok = ans == self.ref
            else:
                from math_verify import parse, verify
                try:
                    ok = bool(verify(self.parsed_ref, parse(f"${ans}$")))
                except Exception:
                    ok = False
            self.cache[ans] = ok
        return self.cache[ans]


def load_records(repo: str) -> list[dict]:
    files = [s.rfilename for s in HfApi().dataset_info(f"{ORG}/{repo}").siblings if s.rfilename.endswith(".tar.gz")]
    path = hf_hub_download(f"{ORG}/{repo}", files[0], repo_type="dataset")
    records, seen = [], set()
    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".yaml") and "/logs/" not in m.name]
        members.sort(key=lambda m: int(Path(m.name).stem) if Path(m.name).stem.isdigit() else m.name)
        for m in members:
            rec = yaml.load(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8"), Loader=yaml.CSafeLoader)
            if rec["question"] in seen:           # MATH128 Llama-3.3-70B repeats one problem
                continue
            seen.add(rec["question"])
            records.append(rec)
    return records


def prepare(name: str, out: Path) -> Path:
    repo, task, keep = SETS[name]
    records = load_records(repo)
    n = min(len(r["samples"]) for r in records)
    n = min(n, keep) if keep else n
    labels = np.empty((len(records), n), dtype=np.int16)
    tokenizer = None
    if name in TOKENIZER:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER[name], "tokenizer.json"))
        tokens = np.empty((len(records), n), dtype=np.int32)
    for q, rec in enumerate(records):
        grade, ids = Grader(task, str(rec["gt_answer"])), {}
        samples = rec["samples"][:n]
        for i, s in enumerate(samples):
            a = extract(task, s)
            if a is None or a == "":
                labels[q, i] = -1
            elif grade(a):
                labels[q, i] = 0
            else:
                labels[q, i] = ids.setdefault(a, len(ids) + 1)
        if tokenizer is not None:
            tokens[q] = [len(e.ids) for e in tokenizer.encode_batch(samples, add_special_tokens=False)]
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{name}.npz"
    arrays = {"labels": labels}
    if tokenizer is not None:
        arrays["tokens"] = tokens
    np.savez_compressed(dest, **arrays)
    (out / f"{name}.json").write_text(json.dumps({"source": f"{ORG}/{repo}", "task": task, "problems": len(records),
                                                  "samples_per_problem": n}, indent=1))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--only", nargs="*", choices=list(SETS))
    args = ap.parse_args()
    for name in args.only or SETS:
        z = np.load(prepare(name, args.out))
        lab = z["labels"]
        extra = f", mean tokens {z['tokens'].mean():.0f}" if "tokens" in z else ""
        print(f"{name}: {lab.shape[0]} problems x {lab.shape[1]} samples, correct {np.mean(lab == 0):.3f}, "
              f"no answer {np.mean(lab == -1):.3f}{extra}")


if __name__ == "__main__":
    main()
