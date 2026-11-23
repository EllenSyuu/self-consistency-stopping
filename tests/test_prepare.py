"""Answer extraction and normalisation used to build the label arrays."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prep", Path(__file__).resolve().parents[1] / "scripts" / "prepare_monkey_business.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_gsm8k_takes_the_last_hash_answer():
    assert prep.extract("GSM8K", "so 3 + 4 = 7\n#### 7\nwait\n#### 1,150") == "1150"
    assert prep.extract("GSM8K", "no final line") is None


def test_math_prefers_final_answer_line_then_last_box():
    s = r"so \boxed{3}. Final Answer: The final answer is $\frac{1}{2}$. I hope it is correct."
    assert prep.extract("MATH", s) == r"\frac{1}{2}"
    assert prep.extract("MATH", r"thus $x=\boxed{\dfrac{a}{\left(b\right)}}$") == r"\frac{a}{(b)}"


def test_normalise_merges_trivial_spellings():
    assert prep.normalise(" 12.50 ") == prep.normalise("12.5") == "12.5"
    assert prep.normalise(r"\text{ apples}") == "apples"


def test_labels_merge_correct_samples_and_flag_missing():
    lab = prep.labels_for("GSM8K", ["#### 5", "#### 6", "x", "#### 6.0"], [True, False, False, False])
    assert lab.tolist() == [0, 1, -1, 1]
