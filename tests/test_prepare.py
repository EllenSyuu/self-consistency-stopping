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


gen = importlib.util.spec_from_file_location(
    "gen", Path(__file__).resolve().parents[1] / "scripts" / "prepare_sc_genrm.py")
genrm = importlib.util.module_from_spec(gen)
gen.loader.exec_module(genrm)


def test_aime_answers_are_integers_from_the_last_box():
    assert genrm.extract("AIME", r"first \boxed{12}, so finally \boxed{070}") == "70"
    assert genrm.Grader("AIME", "70")("70") and not genrm.Grader("AIME", "70")("7")
    assert genrm.extract("AIME", "ran out of tokens before answering") is None


def test_gpqa_takes_the_letter_after_the_final_answer_phrase():
    assert genrm.extract("GPQA", "Final Answer: The final answer is B.") == "B"
    assert genrm.extract("GPQA", r"The final answer is $\boxed{C}$") == "C"
    assert genrm.extract("GPQA", "I think A or D") is None


def test_math_grading_accepts_equivalent_forms():
    g = genrm.Grader("MATH", r"so the answer is $\boxed{\frac{1}{2}}$")
    assert g(genrm.extract("MATH", r"thus $\boxed{0.5}$")) and not g("\\frac{1}{3}")
