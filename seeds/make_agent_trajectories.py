"""Make tool-use training trajectories for Neuro's agents (agents.py), with the agents' REAL tools.

Each trajectory is a whole agent run in the exact format agents.py uses at runtime: its system prompt
for the task kind, the task, the model's tool calls (READ, GLOB, GREP, TODO, EDIT, RUN, CALC), the
real tool results in <tool_response>, and a short final answer. The tools really run on a small
generated project in workspace/agent-train/, so every READ shows real numbered lines, every RUN a
real traceback or "all tests passed", and every CALC a real result. Nothing is made up.

    python seeds/make_agent_trajectories.py --count 3000 --out D:/datasets/agent/train.jsonl
    python seeds/make_agent_trajectories.py --count 150 --seed 99 --out D:/datasets/agent/heldout.jsonl

Kinds: code (fix a bug: read, edit, run the tests; or run first, then read and fix), research
(find a function, read it, answer with file:line), recovery (a guessed path is missing: GLOB, then
READ), math (CALC, then `Answer:`), quick (a plain answer, no tool). About a quarter in Hebrew.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import agents  # noqa: E402

ROOT = agents.ROOT
WORK = "workspace/agent-train"


# ---------------------------------------------------------------- bug templates (python)
# (name stem, correct body, buggy body, tests, what was wrong in English, in Hebrew)
def _bugs(r: random.Random) -> list[tuple]:
    k = r.randint(3, 9)
    return [
        ("sum_to", "def {f}(n):\n    \"\"\"Sum of 1..n.\"\"\"\n    return sum(range(1, n + 1))\n",
         "def {f}(n):\n    \"\"\"Sum of 1..n.\"\"\"\n    return sum(range(1, n))\n",
         ["{f}(1) == 1", "{f}(4) == 10", f"{{f}}({k}) == {k * (k + 1) // 2}", "{f}(0) == 0"],
         "an off-by-one: range(1, n) stopped before n", "טעות של אחד: range(1, n) נעצר לפני n"),
        ("largest", "def {f}(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x > best:\n            best = x\n    return best\n",
         "def {f}(xs):\n    best = xs[0]\n    for x in xs[1:]:\n        if x < best:\n            best = x\n    return best\n",
         ["{f}([3, 9, 2]) == 9", "{f}([-5, -1]) == -1", f"{{f}}([{k}]) == {k}"],
         "the comparison was reversed (x < best), so it returned the smallest", "ההשוואה הייתה הפוכה (x < best) ולכן הוחזר הקטן ביותר"),
        ("mean_of", "def {f}(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)\n",
         "def {f}(xs):\n    return sum(xs) / len(xs)\n",
         ["{f}([2, 4]) == 3", "{f}([]) == 0.0", "{f}([5]) == 5"],
         "an empty list divided by zero; it now returns 0.0", "רשימה ריקה גרמה לחלוקה באפס; עכשיו מוחזר 0.0"),
        ("word_count", "def {f}(text):\n    return len(text.split())\n",
         "def {f}(text):\n    return len(text.split(\" \"))\n",
         ["{f}(\"a b  c\") == 3", "{f}(\"\") == 0", "{f}(\" one \") == 1"],
         "split(\" \") counted empty strings between spaces; split() does not", "split(\" \") ספר מחרוזות ריקות בין רווחים; split() לא"),
        ("is_even", "def {f}(n):\n    return n % 2 == 0\n", "def {f}(n):\n    return n % 2 == 1\n",
         ["{f}(4)", "not {f}(7)", "{f}(0)", "{f}(-2)"],
         "it tested n % 2 == 1, which is odd", "הבדיקה הייתה n % 2 == 1, כלומר אי-זוגי"),
        ("clamp", "def {f}(x, lo, hi):\n    return max(lo, min(x, hi))\n", "def {f}(x, lo, hi):\n    return max(lo, min(x, lo))\n",
         ["{f}(5, 0, 10) == 5", "{f}(-3, 0, 10) == 0", "{f}(42, 0, 10) == 10"],
         "min(x, lo) used the lower bound twice; it must be min(x, hi)", "min(x, lo) השתמש בגבול התחתון פעמיים; צריך min(x, hi)"),
        ("reverse_words", "def {f}(s):\n    return \" \".join(s.split()[::-1])\n", "def {f}(s):\n    return \" \".join(s.split())\n",
         ["{f}(\"a b c\") == \"c b a\"", "{f}(\"one\") == \"one\"", "{f}(\"\") == \"\""],
         "the words were never reversed ([::-1] was missing)", "המילים לא התהפכו (חסר [::-1])"),
        ("percent", "def {f}(part, whole):\n    return part / whole * 100\n", "def {f}(part, whole):\n    return part // whole * 100\n",
         ["{f}(1, 4) == 25", "{f}(3, 3) == 100", "{f}(1, 8) == 12.5"],
         "integer division (//) dropped the fraction; it must be /", "חלוקה בשלמים (//) איבדה את השבר; צריך /"),
        ("last_n", "def {f}(xs, n):\n    if n <= 0:\n        return []\n    return xs[-n:]\n", "def {f}(xs, n):\n    return xs[-n:]\n",
         ["{f}([1, 2, 3], 2) == [2, 3]", "{f}([1, 2, 3], 0) == []", "{f}([1], 5) == [1]"],
         "xs[-0:] returned the whole list for n == 0", "xs[-0:] החזיר את כל הרשימה כש-n == 0"),
        ("add_tag", "def {f}(tag, tags=None):\n    if tags is None:\n        tags = []\n    tags.append(tag)\n    return tags\n",
         "def {f}(tag, tags=[]):\n    tags.append(tag)\n    return tags\n",
         ["{f}(\"a\") == [\"a\"]", "{f}(\"b\") == [\"b\"]", "{f}(\"c\", [\"x\"]) == [\"x\", \"c\"]"],
         "a mutable default list was shared between calls", "רשימה כברירת מחדל נשמרה בין קריאות"),
        ("fizz", "def {f}(n):\n    if n % 15 == 0:\n        return \"FizzBuzz\"\n    if n % 3 == 0:\n        return \"Fizz\"\n    if n % 5 == 0:\n        return \"Buzz\"\n    return str(n)\n",
         "def {f}(n):\n    if n % 3 == 0:\n        return \"Fizz\"\n    if n % 5 == 0:\n        return \"Buzz\"\n    if n % 15 == 0:\n        return \"FizzBuzz\"\n    return str(n)\n",
         ["{f}(15) == \"FizzBuzz\"", "{f}(9) == \"Fizz\"", "{f}(10) == \"Buzz\"", "{f}(7) == \"7\""],
         "the 15 check came after the 3 check, so it was never reached", "בדיקת 15 הגיעה אחרי בדיקת 3 ולכן לא הגיעו אליה"),
        ("total_qty", "def {f}(order):\n    return sum(item[\"quantity\"] for item in order)\n",
         "def {f}(order):\n    return sum(item[\"qty\"] for item in order)\n",
         ["{f}([{{\"quantity\": 2}}, {{\"quantity\": 3}}]) == 5", "{f}([]) == 0"],
         "it read the key \"qty\", but the items use \"quantity\"", "הקוד קרא את המפתח \"qty\" אבל בפריטים יש \"quantity\""),
    ]


HELPERS = [
    "def {g}(a, b):\n    return a * b + 1\n",
    "def {g}(text):\n    return text.strip().title()\n",
    "def {g}(xs):\n    return [x for x in xs if x is not None]\n",
    "def {g}(n):\n    return n * n\n",
]
NAMES = ["util", "core", "tools", "logic", "helpers", "mathx", "orders", "textkit", "stats", "items"]
SUFFIX = "abcdefghjkmnpqrstuvwxyz"


def tag(r: random.Random) -> str:
    return "".join(r.choice(SUFFIX) for _ in range(4))


class Run:
    """Builds one trajectory: the system prompt and task, then call/result pairs, then the answer."""

    def __init__(self, kind: str, task: str) -> None:
        self.kind = kind
        self.state = agents.TaskState(task)
        self.messages = [
            {"role": "system", "content": agents.agent_system(kind)},
            {"role": "user", "content": "Task:\n" + task + "\n\n" + agents.workspace_note()},
        ]

    def call(self, *commands: str) -> list[str]:
        results = []
        for command in commands:
            result, _made = agents.run_tool(command, command, False, kind=self.kind, state=self.state)
            results.append(result)
        self.messages.append({"role": "assistant", "content": "\n".join(commands)})
        self.messages.append({"role": "user", "content": "\n".join(agents.tool_response(x) for x in results)})
        return results

    def answer(self, text: str) -> dict:
        self.messages.append({"role": "assistant", "content": text.strip()})
        return {"messages": self.messages, "kind": self.kind, "source": "agent-trajectories-v1"}


def project(r: random.Random, folder: Path, he: bool):
    bug = r.choice(_bugs(r))
    stem, good, bad, tests, why_en, why_he = bug
    f = f"{stem}_{tag(r)}"
    module = r.choice(NAMES) + "_" + tag(r)
    helpers = [h.format(g=f"helper_{tag(r)}") for h in r.sample(HELPERS, r.randint(1, 2))]
    parts = helpers[:1] + [bad.format(f=f)] + helpers[1:]
    source = "\n\n".join(parts)
    lines = [f"from {module} import {f}", ""]
    asserts = [t.format(f=f) for t in tests]
    lines += [f"assert {a}, {json.dumps(a)}" for a in asserts]
    lines += ["", f"print(\"all {len(asserts)} tests passed\")", ""]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{module}.py").write_text(source, encoding="utf-8")
    (folder / f"test_{module}.py").write_text("\n".join(lines), encoding="utf-8")
    return f, module, bad.format(f=f), good.format(f=f), len(asserts), why_he if he else why_en


def code_task(r: random.Random, idx: int, he: bool) -> dict | None:
    rel = f"{WORK}/t{idx}"
    folder = ROOT / rel
    f, module, bad, good, count, why = project(r, folder, he)
    mod, test = f"{rel}/{module}.py", f"{rel}/test_{module}.py"
    if he:
        task = f"הטסטים ב-{test} נכשלים. תמצא את הבאג ב-{mod} ותתקן אותו."
    else:
        task = f"The tests in {test} fail. Find the bug in {mod} and fix it."
    run = Run("code", task)
    # Run with the test folder on the path, the way a user would.
    run_cmd = f"RUN cd {rel} && python test_{module}.py"
    if r.random() < 0.5:
        first = run.call(run_cmd)[0]
        if "exit 0" in first or "passed" in first:
            return None
        run.call(f"READ {mod}")
    else:
        run.call("TODO\n[>] Read the test and the module\n[ ] Fix the bug\n[ ] Run the tests", f"READ {test}", f"READ {mod}")
    old_lines = [line for line, new in zip(bad.splitlines(), good.splitlines()) if line != new]
    if len(bad.splitlines()) == len(good.splitlines()) and old_lines:
        pairs = [(a, b) for a, b in zip(bad.splitlines(), good.splitlines()) if a != b]
        old, new = "\n".join(a for a, _ in pairs), "\n".join(b for _, b in pairs)
    else:
        old, new = bad.rstrip("\n"), good.rstrip("\n")
    edited = run.call(agents.edit_call(mod, old, new))[0]
    if not edited.startswith("edited"):
        return None
    result = run.call(run_cmd)[0]
    if "passed" not in result:
        return None
    if he:
        text = f"תיקנתי את {f} ב-{mod}: {why}.\nבדיקה: הרצתי את {test} והתוצאה: כל {count} הטסטים עוברים."
    else:
        text = f"Fixed {f} in {mod}: {why}.\nChecked: ran {test}; all {count} tests pass."
    return run.answer(text)


def research_task(r: random.Random, idx: int, he: bool) -> dict | None:
    rel = f"{WORK}/r{idx}"
    folder = ROOT / rel
    f, module, bad, good, _count, _why = project(r, folder, he)
    (folder / f"{module}.py").write_text((folder / f"{module}.py").read_text(encoding="utf-8").replace(bad, good), encoding="utf-8")
    probe = {"sum_to": "5", "largest": "[4, 11, 7]", "mean_of": "[1, 2, 6]", "word_count": "\"to be or not\"",
             "is_even": "10", "clamp": "15, 0, 9", "reverse_words": "\"left right\"", "percent": "3, 12",
             "last_n": "[5, 6, 7, 8], 3", "add_tag": "\"x\"", "fizz": "30", "total_qty": "[{\"quantity\": 4}]"}[f.rsplit("_", 1)[0]]
    scope: dict = {}
    exec(compile(good, "<research>", "exec"), scope)  # our own template code
    value = eval(f"{f}({probe})", scope)  # noqa: S307 - our own template call
    if he:
        task = f"איפה מוגדרת הפונקציה {f} בתיקייה {rel}, ומה היא מחזירה עבור {f}({probe})?"
    else:
        task = f"Where is the function {f} defined under {rel}, and what does {f}({probe}) return?"
    run = Run("research", task)
    found = run.call(f"GREP def {f}")[0]
    if f"{module}.py" not in found:
        return None
    line_no = next((int(part) for part in found.split(":")[1:2] if part.strip().isdigit()), None)
    read = run.call(f"READ {rel}/{module}.py")[0]
    if line_no is None:
        line_no = next((int(row.split(":", 1)[0]) for row in read.splitlines() if f"def {f}" in row and row.split(":", 1)[0].strip().isdigit()), 1)
    if he:
        text = f"`{f}` מוגדרת ב-{rel}/{module}.py:{line_no}.\n{f}({probe}) מחזירה {value!r}."
    else:
        text = f"`{f}` is defined in {rel}/{module}.py:{line_no}.\n{f}({probe}) returns {value!r}."
    return run.answer(text)


def recovery_task(r: random.Random, idx: int, he: bool) -> dict | None:
    rel = f"{WORK}/c{idx}"
    folder = ROOT / rel
    folder.mkdir(parents=True, exist_ok=True)
    key, value = r.choice([("timeout_s", r.randint(5, 120)), ("max_workers", r.randint(2, 32)), ("retries", r.randint(1, 9)),
                           ("port", r.choice([8000, 8081, 9090, 5173]))])
    real = f"settings_{tag(r)}.py"
    (folder / real).write_text(f"# app settings\nname = \"demo\"\n{key} = {value}\ndebug = False\n", encoding="utf-8")
    if he:
        task = f"מה הערך של {key} בקובץ ההגדרות שבתיקייה {rel}?"
    else:
        task = f"What is {key} set to in the settings file under {rel}?"
    run = Run("research", task)
    first = run.call(f"READ {rel}/config.py")[0]
    if first.strip() != "missing":
        return None
    listed = run.call(f"GLOB {rel}/*.py")[0]
    if real not in listed:
        return None
    run.call(f"READ {rel}/{real}")
    text = (f"{key} = {value} (ב-{rel}/{real}:3; אין קובץ config.py)." if he
            else f"{key} is {value} ({rel}/{real}:3; there is no config.py).")
    return run.answer(text)


def math_task(r: random.Random, he: bool) -> dict | None:
    kind = r.randrange(8)
    if kind == 0:
        a, b, c, d = (r.randint(2, 99) for _ in range(4))
        q = f"חשב את ({a} + {b}) * {c} - {d} / 4 בדיוק." if he else f"Compute ({a} + {b}) * {c} - {d} / 4 exactly."
        code = f"from fractions import Fraction as F\nprint(F({a} + {b}) * {c} - F({d}, 4))"
    elif kind == 1:
        price, up, down = r.randrange(100, 3000, 10), r.choice([10, 15, 20, 25]), r.choice([10, 20, 30])
        q = (f"מחיר של {price} עולה ב-{up}% ואז יורד ב-{down}%. מה המחיר הסופי?" if he
             else f"A price of {price} rises {up}% and then falls {down}%. What is the final price?")
        code = f"from fractions import Fraction as F\nprint(F({price}) * F(100 + {up}, 100) * F(100 - {down}, 100))"
    elif kind == 2:
        a, b, c = r.randint(2, 9), r.randint(-20, 20), r.randint(-40, 40)
        q = f"{a}x + {b} = {c}".replace("+ -", "- ")
        q = f"פתור: {q}" if he else f"Solve {q} for x."
        code = f"from fractions import Fraction as F\nx = F({c} - ({b}), {a})\nprint(x, {a} * x + {b} == {c})"
    elif kind == 3:
        n, k = r.randint(8, 30), r.randint(2, 6)
        q = (f"בכמה דרכים אפשר לבחור {k} מתוך {n}?" if he else f"How many ways are there to choose {k} of {n}?")
        code = f"import math\nprint(math.comb({n}, {k}))"
    elif kind == 4:
        t = r.randint(3, 11)
        q = (f"מה ההסתברות שסכום שתי קוביות הוא {t}?" if he else f"What is the probability that two dice sum to {t}?")
        code = f"from fractions import Fraction as F\nhits = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == {t})\nprint(F(hits, 36))"
    elif kind == 5:
        a, e, m = r.randint(2, 50), r.randint(20, 400), r.choice([7, 13, 97, 1000])
        q = (f"מה {a}^{e} מודולו {m}?" if he else f"What is {a}^{e} mod {m}?")
        code = f"print(pow({a}, {e}, {m}))"
    elif kind == 6:
        raw = r.randint(0x80000000, 0xFFFFFFFF)
        q = (f"מה הערך של 0x{raw:08X} כ-int32 עם סימן?" if he else f"What is 0x{raw:08X} as a signed int32?")
        code = f"v = 0x{raw:08X}\nprint(v - (1 << 32) if v >= 1 << 31 else v)"
    else:
        data = [r.randint(0, 255) for _ in range(4)]
        shown = " ".join(f"{b:02X}" for b in data)
        q = (f"הבתים {shown} (little-endian) כ-uint32: מה הערך העשרוני?" if he else f"The bytes {shown} (little-endian) as a uint32: what is the decimal value?")
        code = f"print(int.from_bytes(bytes([{', '.join(f'0x{b:02X}' for b in data)}]), 'little'))"
    run = Run("math", q)
    out = run.call("CALC\n" + code + "\nENDCALC")[0].strip()
    if not out or "exit" in out or "blocked" in out or "Error" in out:
        return None
    value = out.split()[0]
    text = (f"חישבתי עם CALC: {out}.\nAnswer: {value}" if he else f"Computed with CALC: {out}.\nAnswer: {value}")
    return run.answer(text)


QUICK = [
    ("say hello", "Hello! What would you like to work on?"),
    ("תגיד שלום", "שלום! במה אפשר לעזור?"),
    ("what can you do?", "I can read and search this repo, fix code and run its tests, answer questions about the code, and compute math with CALC."),
    ("מה אתה יודע לעשות?", "אני יכול לקרוא ולחפש בקוד, לתקן באגים ולהריץ טסטים, לענות על שאלות על הקוד, ולחשב מתמטיקה עם CALC."),
]


def quick_task(r: random.Random) -> dict:
    task, reply = r.choice(QUICK)
    run = Run("quick", task)
    return run.answer(reply)


def build(count: int, seed: int, he_share: float) -> list[dict]:
    r = random.Random(seed)
    work = ROOT / WORK
    rows: list[dict] = []
    tries = 0
    try:
        while len(rows) < count and tries < count * 3:
            tries += 1
            he = r.random() < he_share
            pick = r.random()
            if pick < 0.45:
                name, row = "code", code_task(r, tries, he)
            elif pick < 0.65:
                name, row = "research", research_task(r, tries, he)
            elif pick < 0.75:
                name, row = "recovery", recovery_task(r, tries, he)
            elif pick < 0.97:
                name, row = "math", math_task(r, he)
            else:
                name, row = "quick", quick_task(r)
            if row:
                row["type"] = name
                row["hl"] = "he" if he else "en"
                rows.append(row)
            if tries % 50 == 0:
                shutil.rmtree(work, ignore_errors=True)
                print(f"  {len(rows)} trajectories ({tries} tries)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hebrew", type=float, default=0.25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = build(args.count, args.seed, args.hebrew)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["type"]] = kinds.get(row["type"], 0) + 1
    print(f"wrote {len(rows)} trajectories to {out}: {kinds}")


if __name__ == "__main__":
    main()
