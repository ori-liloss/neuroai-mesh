"""Make verified math tasks for the mesh: every answer is computed here, in Python, so the
verify_math check knows the right answer and only correct solutions become training data.

    python seeds/make_math_seeds.py                # writes seeds/math-v1.jsonl (1200 tasks)
    python seeds/make_math_seeds.py --count 3000 --seed 7 --out seeds/math-v2.jsonl
    python mesh.py jobs add generate-verify seeds/math-v1.jsonl

Topics: arithmetic, fractions, percents, equations, systems, quadratics, number theory,
combinatorics, probability, sequences, geometry, statistics, word problems (English and
Hebrew), and the math of reverse engineering: two's complement, hex, bit operations, byte
order, C struct layout, address arithmetic, x86 register traces. Plus "what does this
Python program print", which trains careful step-by-step tracing of code.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "seeds/math-v1"

ASK_EN = "\n\nSolve it step by step and check your result. Put the final answer alone in \\boxed{...}."
ASK_HE = "\n\nפתור שלב אחר שלב ובדוק את התוצאה. כתוב את התשובה הסופית לבדה בתוך \\boxed{...}."
FRACTION_NOTE_EN = " Give an exact answer (an integer or a fraction like 3/4)."
FRACTION_NOTE_HE = " תן תשובה מדויקת (מספר שלם או שבר כמו 3/4)."


def fmt(value: Fraction | int) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------- general math


def arithmetic(r: random.Random, he: bool):
    a, b, c, d = (r.randint(2, 60) for _ in range(4))
    e = r.randint(2, 9)
    shape = r.randrange(3)
    if shape == 0:
        text, value = f"({a} + {b}) * {c} - {d} / {e}", Fraction(a + b) * c - Fraction(d, e)
    elif shape == 1:
        text, value = f"{a} * {b} - {c} * {d} + {e}^3", Fraction(a * b - c * d + e ** 3)
    else:
        text, value = f"({a} - {b})^2 / {e} + {c}", Fraction((a - b) ** 2, e) + c
    q = (f"חשב את הביטוי {text}." + FRACTION_NOTE_HE) if he else (f"Compute {text}." + FRACTION_NOTE_EN)
    return "arithmetic", q, fmt(value)


def fractions_task(r: random.Random, he: bool):
    x = Fraction(r.randint(1, 9), r.randint(2, 12))
    y = Fraction(r.randint(1, 9), r.randint(2, 12))
    z = Fraction(r.randint(1, 9), r.randint(2, 12))
    value = (x + y) * z - x / (y + 1)
    text = f"({x} + {y}) * {z} - {x} / ({y} + 1)"
    q = (f"חשב: {text}." + FRACTION_NOTE_HE) if he else (f"Compute {text}." + FRACTION_NOTE_EN)
    return "fractions", q, fmt(value)


def percent(r: random.Random, he: bool):
    price = r.randrange(80, 2000, 10)
    up = r.choice([5, 10, 12, 15, 20, 25, 30])
    down = r.choice([5, 10, 15, 20, 25, 40, 50])
    value = Fraction(price) * (100 + up) / 100 * (100 - down) / 100
    if he:
        q = (f"מחיר של מוצר הוא {price} ש\"ח. המחיר עלה ב-{up}%, ואחר כך ירד ב-{down}% מהמחיר החדש. "
             "מה המחיר הסופי בש\"ח?" + FRACTION_NOTE_HE)
    else:
        q = (f"A product costs {price} dollars. The price rises by {up}%, then falls by {down}% of the new price. "
             "What is the final price in dollars?" + FRACTION_NOTE_EN)
    return "percent", q, fmt(value)


def linear(r: random.Random, he: bool):
    while True:
        a, c = r.randint(-9, 12), r.randint(-9, 12)
        if a != c and a and c:
            break
    x = Fraction(r.randint(-20, 20), r.choice([1, 1, 1, 2, 3]))
    b = r.randint(-30, 30)
    d = a * x + b - c * x
    eq = f"{a}x + {b} = {c}x + {fmt(d)}".replace("+ -", "- ")
    q = (f"פתור את המשוואה {eq}. מה הערך של x?" + FRACTION_NOTE_HE) if he else (f"Solve {eq} for x." + FRACTION_NOTE_EN)
    return "linear_equation", q, fmt(x)


def system2(r: random.Random, he: bool):
    x, y = r.randint(-12, 12), r.randint(-12, 12)
    while True:
        a, b, c, d = (r.randint(-7, 7) for _ in range(4))
        if a * d - b * c != 0 and 0 not in (a, b, c, d):
            break
    e, f = a * x + b * y, c * x + d * y
    eqs = f"{a}x + {b}y = {e} and {c}x + {d}y = {f}".replace("+ -", "- ")
    q = (f"נתונה מערכת המשוואות {eqs.replace(' and ', ' וגם ')}. מה הערך של x + 2y?") if he else (
        f"Solve the system {eqs}. What is x + 2y?")
    return "linear_system", q, str(x + 2 * y)


def quadratic(r: random.Random, he: bool):
    p, q_ = r.randint(-15, 15), r.randint(-15, 15)
    k = r.choice([1, 1, 2, 3])
    b, c = -k * (p + q_), k * p * q_
    eq = f"{k}x^2 + {b}x + {c} = 0".replace("+ -", "- ")
    ask = r.choice(["larger", "product"])
    if ask == "larger":
        answer = max(p, q_)
        q = (f"מצא את הפתרון הגדול של המשוואה {eq}.") if he else (f"Find the larger root of {eq}.")
    else:
        answer = p * q_
        q = (f"מה מכפלת הפתרונות של המשוואה {eq}?") if he else (f"What is the product of the roots of {eq}?")
    return "quadratic", q, str(answer)


def number_theory(r: random.Random, he: bool):
    kind = r.randrange(4)
    if kind == 0:
        a, b = r.randint(12, 900), r.randint(12, 900)
        q = (f"מה המחלק המשותף הגדול ביותר של {a} ו-{b}?") if he else (f"What is gcd({a}, {b})?")
        return "gcd", q, str(math.gcd(a, b))
    if kind == 1:
        a, b = r.randint(4, 60), r.randint(4, 60)
        q = (f"מה הכפולה המשותפת הקטנה ביותר של {a} ו-{b}?") if he else (f"What is lcm({a}, {b})?")
        return "lcm", q, str(math.lcm(a, b))
    if kind == 2:
        a, e, m = r.randint(2, 40), r.randint(10, 200), r.choice([7, 11, 13, 17, 19, 23, 97, 100, 1000])
        q = (f"מה השארית של {a}^{e} בחלוקה ב-{m}?") if he else (f"What is {a}^{e} mod {m}?")
        return "modpow", q, str(pow(a, e, m))
    n = r.randint(50, 400)
    count = sum(1 for k in range(2, n + 1) if all(k % p for p in range(2, int(k ** 0.5) + 1)))
    q = (f"כמה מספרים ראשוניים יש בין 1 ל-{n} (כולל)?") if he else (f"How many primes are there from 1 to {n} inclusive?")
    return "primes", q, str(count)


def combinatorics(r: random.Random, he: bool):
    kind = r.randrange(3)
    if kind == 0:
        n, k = r.randint(6, 20), r.randint(2, 6)
        q = (f"בכמה דרכים אפשר לבחור ועדה של {k} אנשים מתוך {n}?") if he else (
            f"In how many ways can a committee of {k} be chosen from {n} people?")
        return "combinations", q, str(math.comb(n, k))
    if kind == 1:
        n, k = r.randint(5, 12), r.randint(2, 4)
        q = (f"בכמה דרכים אפשר לחלק {k} פרסים שונים (ראשון, שני ...) בין {n} משתתפים, כל משתתף לכל היותר פרס אחד?") if he else (
            f"In how many ways can {k} different prizes (first, second, ...) go to {n} contestants, at most one prize each?")
        return "permutations", q, str(math.perm(n, k))
    word = r.choice(["LETTER", "BALLOON", "MISSISSIPPI", "BANANA", "COMMITTEE", "SUCCESS"])
    counts: dict[str, int] = {}
    for ch in word:
        counts[ch] = counts.get(ch, 0) + 1
    value = math.factorial(len(word))
    for c in counts.values():
        value //= math.factorial(c)
    q = (f"כמה סידורים שונים יש לאותיות של המילה {word}?") if he else (
        f"How many distinct arrangements are there of the letters of {word}?")
    return "multiset_permutations", q, str(value)


def probability(r: random.Random, he: bool):
    kind = r.randrange(3)
    if kind == 0:
        target = r.randint(4, 10)
        hits = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == target)
        q = (f"זורקים שתי קוביות הוגנות. מה ההסתברות שסכומן יהיה {target}?" + FRACTION_NOTE_HE) if he else (
            f"Two fair dice are rolled. What is the probability that the sum is {target}?" + FRACTION_NOTE_EN)
        return "dice", q, fmt(Fraction(hits, 36))
    if kind == 1:
        n, k = r.randint(4, 10), None
        k = r.randint(1, n - 1)
        q = (f"מטילים מטבע הוגן {n} פעמים. מה ההסתברות לקבל בדיוק {k} פעמים עץ?" + FRACTION_NOTE_HE) if he else (
            f"A fair coin is flipped {n} times. What is the probability of exactly {k} heads?" + FRACTION_NOTE_EN)
        return "binomial", q, fmt(Fraction(math.comb(n, k), 2 ** n))
    red, blue = r.randint(2, 9), r.randint(2, 9)
    q = (f"בשקית {red} כדורים אדומים ו-{blue} כחולים. מוציאים שני כדורים בלי החזרה. מה ההסתברות ששניהם אדומים?"
         + FRACTION_NOTE_HE) if he else (
        f"A bag holds {red} red and {blue} blue balls. Two are drawn without replacement. "
        "What is the probability that both are red?" + FRACTION_NOTE_EN)
    return "draws", q, fmt(Fraction(red, red + blue) * Fraction(red - 1, red + blue - 1))


def sequences(r: random.Random, he: bool):
    if r.random() < 0.5:
        a, d, n = r.randint(-20, 30), r.randint(-7, 9) or 3, r.randint(10, 60)
        total = n * (2 * a + (n - 1) * d) // 2
        q = (f"בסדרה חשבונית האיבר הראשון הוא {a} וההפרש הוא {d}. מה סכום {n} האיברים הראשונים?") if he else (
            f"An arithmetic sequence starts at {a} with common difference {d}. What is the sum of the first {n} terms?")
        return "arithmetic_series", q, str(total)
    a, ratio, n = r.randint(1, 9), r.choice([2, 3, -2]), r.randint(5, 12)
    total = sum(a * ratio ** i for i in range(n))
    q = (f"בסדרה הנדסית האיבר הראשון הוא {a} והמנה היא {ratio}. מה סכום {n} האיברים הראשונים?") if he else (
        f"A geometric sequence starts at {a} with ratio {ratio}. What is the sum of the first {n} terms?")
    return "geometric_series", q, str(total)


TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41)]


def geometry(r: random.Random, he: bool):
    kind = r.randrange(3)
    if kind == 0:
        a, b, c = r.choice(TRIPLES)
        k = r.randint(1, 6)
        q = (f"במשולש ישר זווית הניצבים הם {a * k} ו-{b * k}. מה אורך היתר?") if he else (
            f"A right triangle has legs {a * k} and {b * k}. How long is the hypotenuse?")
        return "pythagoras", q, str(c * k)
    if kind == 1:
        w, h, cut = r.randint(5, 30), r.randint(5, 30), r.randint(1, 4)
        area = w * h - cut * cut * 4
        q = (f"ממלבן בגודל {w} על {h} חותכים ריבוע של {cut} על {cut} מכל אחת מארבע הפינות. מה השטח שנשאר?") if he else (
            f"From a {w} by {h} rectangle a {cut} by {cut} square is cut out of each of the four corners. What area is left?")
        return "area", q, str(area)
    w, h, cut = r.randint(10, 30), r.randint(10, 30), r.randint(1, 4)
    volume = (w - 2 * cut) * (h - 2 * cut) * cut
    q = (f"מקרטון בגודל {w} על {h} חותכים ריבוע {cut} על {cut} בכל פינה ומקפלים לקופסה פתוחה. מה נפח הקופסה?") if he else (
        f"A {w} by {h} sheet has a {cut} by {cut} square cut from each corner and is folded into an open box. "
        "What is the volume of the box?")
    return "volume", q, str(volume)


def statistics_task(r: random.Random, he: bool):
    values = [r.randint(1, 99) for _ in range(r.choice([5, 6, 7, 8, 9]))]
    if r.random() < 0.5:
        q = (f"מה הממוצע של המספרים {', '.join(map(str, values))}?" + FRACTION_NOTE_HE) if he else (
            f"What is the mean of {', '.join(map(str, values))}?" + FRACTION_NOTE_EN)
        return "mean", q, fmt(Fraction(sum(values), len(values)))
    s = sorted(values)
    n = len(s)
    median = Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)
    q = (f"מה החציון של המספרים {', '.join(map(str, values))}?" + FRACTION_NOTE_HE) if he else (
        f"What is the median of {', '.join(map(str, values))}?" + FRACTION_NOTE_EN)
    return "median", q, fmt(median)


def word_problem(r: random.Random, he: bool):
    kind = r.randrange(3)
    if kind == 0:
        v1, v2, gap = r.randint(40, 90), r.randint(50, 120), r.randint(60, 600)
        value = Fraction(gap, v1 + v2)
        q = (f"שתי רכבות יוצאות זו לקראת זו מתחנות שהמרחק ביניהן {gap} ק\"מ, במהירויות {v1} ו-{v2} קמ\"ש. "
             "אחרי כמה שעות הן נפגשות?" + FRACTION_NOTE_HE) if he else (
            f"Two trains start towards each other from stations {gap} km apart, at {v1} and {v2} km/h. "
            "After how many hours do they meet?" + FRACTION_NOTE_EN)
        return "meeting", q, fmt(value)
    if kind == 1:
        a, b = r.randint(2, 12), r.randint(2, 12)
        value = Fraction(a * b, a + b)
        q = (f"ברז אחד ממלא בריכה ב-{a} שעות ושני ב-{b} שעות. בכמה שעות ימלאו אותה יחד?" + FRACTION_NOTE_HE) if he else (
            f"One tap fills a pool in {a} hours and another in {b} hours. How long do they take together, in hours?"
            + FRACTION_NOTE_EN)
        return "work_rate", q, fmt(value)
    kids, extra = r.randint(3, 9), r.randint(1, 5)
    age = r.randint(5, 15)
    total = kids * age + extra * kids * (kids - 1) // 2
    q = (f"ל{kids} אחים יש הפרש של {extra} שנים בין כל שני אחים סמוכים. הצעיר בן {age}. מה סכום הגילים?") if he else (
        f"{kids} siblings are each {extra} years apart. The youngest is {age}. What is the sum of their ages?")
    return "ages", q, str(total)


# ---------------------------------------------------------------- reverse engineering math


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def twos_complement(r: random.Random, he: bool):
    bits = r.choice([8, 16, 32])
    value = -r.randint(1, (1 << (bits - 1)) - 1)
    raw = value & ((1 << bits) - 1)
    hexed = f"0x{raw:0{bits // 4}X}"
    q = (f"הערך {hexed} נקרא מהזיכרון כמספר שלם עם סימן ברוחב {bits} ביט (two's complement). מה ערכו העשרוני?") if he else (
        f"The value {hexed} is read as a signed {bits}-bit integer (two's complement). What is its decimal value?")
    return "twos_complement", q, str(value)


def hex_convert(r: random.Random, he: bool):
    value = r.randint(256, 0xFFFFFF)
    if r.random() < 0.5:
        q = (f"המר את 0x{value:X} לעשרוני.") if he else (f"Convert 0x{value:X} to decimal.")
        return "hex_to_dec", q, str(value)
    q = (f"המר את {value} להקסדצימלי. כתוב את התשובה כך: 0x ואחריו ספרות באותיות קטנות, בלי אפסים מובילים.") if he else (
        f"Convert {value} to hexadecimal. Write it as 0x followed by lowercase digits, no leading zeros.")
    return "dec_to_hex", q, f"0x{value:x}"


def bit_ops(r: random.Random, he: bool):
    x = r.randint(0, 0xFFFFFFFF)
    kind = r.randrange(4)
    if kind == 0:
        shift, mask = r.randint(1, 20), r.choice([0xFF, 0xFFF, 0xFFFF, 0x3F, 0x1F])
        expr, value = f"(0x{x:08X} >> {shift}) & 0x{mask:X}", (x >> shift) & mask
    elif kind == 1:
        y = r.randint(0, 0xFFFFFFFF)
        expr, value = f"(0x{x:08X} ^ 0x{y:08X}) & 0xFFFF", (x ^ y) & 0xFFFF
    elif kind == 2:
        n = r.randint(1, 31)
        expr, value = f"rotl32(0x{x:08X}, {n})", u32((x << n) | (x >> (32 - n)))
    else:
        expr, value = f"popcount(0x{x:08X})", bin(x).count("1")
    note = "rotl32 מסובב 32 ביט שמאלה ו-popcount סופר ביטים דולקים. " if he else "rotl32 rotates 32 bits left; popcount counts set bits. "
    q = (f"{note}מה הערך העשרוני של {expr} (כמספר בלי סימן)?") if he else (
        f"{note}What is the unsigned decimal value of {expr}?")
    return "bit_ops", q, str(value)


def endianness(r: random.Random, he: bool):
    data = [r.randint(0, 255) for _ in range(4)]
    shown = " ".join(f"{b:02X}" for b in data)
    value = int.from_bytes(bytes(data), "little")
    if r.random() < 0.5:
        q = (f"בזיכרון של מעבד x86 (little-endian) יושבים הבתים {shown} בכתובות עולות. מה הערך העשרוני של ה-uint32 שהם מייצגים?") if he else (
            f"On x86 (little-endian), memory holds the bytes {shown} at increasing addresses. "
            "What is the decimal value of the uint32 they form?")
        return "little_endian", q, str(value)
    q = (f"הבתים {shown} בזיכרון x86 נקראים כ-int32 עם סימן. מה הערך העשרוני?") if he else (
        f"The bytes {shown} in x86 memory are read as a signed int32. What is its decimal value?")
    return "little_endian_signed", q, str(s32(value))


C_TYPES = [("char", 1), ("short", 2), ("int", 4), ("float", 4), ("long long", 8), ("double", 8), ("void*", 8)]


def struct_layout(r: random.Random, he: bool):
    fields = [r.choice(C_TYPES) for _ in range(r.randint(3, 6))]
    names = [f"f{i}" for i in range(len(fields))]
    offset, offsets, align = 0, [], 1
    for _, size in fields:
        offset = (offset + size - 1) // size * size
        offsets.append(offset)
        offset += size
        align = max(align, size)
    total = (offset + align - 1) // align * align
    body = " ".join(f"{t} {n};" for (t, _), n in zip(fields, names))
    rule = ("x64 (Windows), יישור טבעי, בלי #pragma pack" if he else "x64 (Windows), natural alignment, no #pragma pack")
    if r.random() < 0.5:
        q = (f"ב-{rule}: מה sizeof(struct S) עבור struct S {{ {body} }};?") if he else (
            f"On {rule}: what is sizeof(struct S) for struct S {{ {body} }};?")
        return "struct_sizeof", q, str(total)
    pick = r.randrange(1, len(fields))
    q = (f"ב-{rule}: מה ה-offset בבתים של השדה {names[pick]} ב-struct S {{ {body} }};?") if he else (
        f"On {rule}: what is the byte offset of field {names[pick]} in struct S {{ {body} }};?")
    return "struct_offset", q, str(offsets[pick])


def address_math(r: random.Random, he: bool):
    base = r.randrange(0x140000000, 0x140FFFFFF, 0x10)
    ctype, size = r.choice([("int", 4), ("short", 2), ("double", 8), ("struct of 24 bytes", 24), ("char", 1)])
    index = r.randint(1, 500)
    value = base + index * size
    q = (f"מערך של {ctype} מתחיל בכתובת 0x{base:X}. מה הכתובת של האיבר באינדקס {index}? כתוב 0x ואחריו ספרות באותיות קטנות.") if he else (
        f"An array of {ctype} starts at 0x{base:X}. What is the address of element {index}? "
        "Write 0x followed by lowercase digits.")
    return "address", q, f"0x{value:x}"


def asm_trace(r: random.Random, he: bool):
    regs = {"eax": r.randint(0, 200), "ebx": r.randint(0, 200), "ecx": r.randint(1, 31)}
    lines = [f"mov eax, {regs['eax']}", f"mov ebx, {regs['ebx']}", f"mov ecx, {regs['ecx']}"]
    for _ in range(r.randint(4, 8)):
        op = r.choice(["add", "sub", "xor", "and", "or", "shl", "shr", "imul", "lea", "neg", "not"])
        dst = r.choice(["eax", "ebx"])
        if op in ("add", "sub", "xor", "and", "or"):
            src = r.choice(["eax", "ebx", "ecx", str(r.randint(1, 255))])
            v = regs[src] if src in regs else int(src)
            regs[dst] = u32({"add": regs[dst] + v, "sub": regs[dst] - v, "xor": regs[dst] ^ v,
                             "and": regs[dst] & v, "or": regs[dst] | v}[op])
            lines.append(f"{op} {dst}, {src}")
        elif op in ("shl", "shr"):
            n = r.randint(1, 8)
            regs[dst] = u32(regs[dst] << n) if op == "shl" else regs[dst] >> n
            lines.append(f"{op} {dst}, {n}")
        elif op == "imul":
            n = r.randint(2, 9)
            regs[dst] = u32(s32(regs[dst]) * n)
            lines.append(f"imul {dst}, {dst}, {n}")
        elif op == "lea":
            scale, disp = r.choice([1, 2, 4, 8]), r.randint(0, 64)
            regs[dst] = u32(regs["eax"] + regs["ebx"] * scale + disp)
            lines.append(f"lea {dst}, [eax + ebx*{scale} + {disp}]")
        elif op == "neg":
            regs[dst] = u32(-regs[dst])
            lines.append(f"neg {dst}")
        else:
            regs[dst] = u32(~regs[dst])
            lines.append(f"not {dst}")
    code = "\n".join(lines)
    signed = r.random() < 0.5
    want = s32(regs["eax"]) if signed else regs["eax"]
    how = ("כמספר עם סימן (int32)" if signed else "כמספר בלי סימן (uint32)") if he else (
        "as a signed int32" if signed else "as an unsigned uint32")
    q = (f"מה הערך של eax בסוף קטע הקוד הזה (x86, 32 ביט), {how}?\n\n```asm\n{code}\n```") if he else (
        f"What is the value of eax after this x86 (32-bit) code, {how}?\n\n```asm\n{code}\n```")
    return "asm_trace", q, str(want)


# ---------------------------------------------------------------- what does this program print


def program_output(r: random.Random, he: bool):
    n = r.randint(5, 14)
    k = r.randint(2, 5)
    templates = [
        f"total = 0\nfor i in range(1, {n}):\n    if i % {k} == 0:\n        total += i * i\n    else:\n        total -= i\nprint(total)",
        f"xs = [x * {k} - {n} for x in range({n})]\nprint(sum(x for x in xs if x > 0), len([x for x in xs if x < 0]))",
        f"def f(n):\n    return 1 if n <= 1 else f(n - 1) + f(n - 2) * {k % 3 + 1}\nprint(f({n % 11 + 3}))",
        f"s = '{r.choice(['reverse', 'decompile', 'assembly', 'ghidra', 'register', 'pointer'])}'\n"
        f"print(s[::-1][{k % 4}:] + s[:{k}].upper())",
        f"counts = {{}}\nfor ch in '{r.choice(['mississippi', 'abracadabra', 'bookkeeper', 'committee'])}':\n"
        "    counts[ch] = counts.get(ch, 0) + 1\nprint(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:2])",
        f"a, b = {n}, {k * 7}\nwhile b:\n    a, b = b, a % b\nprint(a, {n} * {k * 7} // a)",
        f"x = {n * 37}\nbits = []\nwhile x:\n    bits.append(x & 1)\n    x >>= 1\nprint(''.join(map(str, reversed(bits))), len(bits))",
        f"stack = []\nfor t in '{n} {k} + 3 * 2 -'.split():\n    if t.isdigit():\n        stack.append(int(t))\n    else:\n"
        "        b, a = stack.pop(), stack.pop()\n        stack.append(a + b if t == '+' else a - b if t == '-' else a * b)\nprint(stack)",
    ]
    code = r.choice(templates)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<seed>", "exec"), {})  # our own fixed templates only
    printed = buffer.getvalue().strip()
    if "\n" in printed or not printed:
        return None
    q = (f"מה התוכנית הזאת ב-Python מדפיסה? עקוב אחריה צעד אחר צעד. בתוך \\boxed{{...}} כתוב בדיוק את הפלט.\n\n```python\n{code}\n```") if he else (
        f"What does this Python program print? Trace it step by step. Put exactly the printed output in \\boxed{{...}}.\n\n```python\n{code}\n```")
    return "program_output", q, printed


GENERATORS = [
    (arithmetic, 5), (fractions_task, 5), (percent, 5), (linear, 6), (system2, 5), (quadratic, 5),
    (number_theory, 6), (combinatorics, 6), (probability, 7), (sequences, 4), (geometry, 5),
    (statistics_task, 3), (word_problem, 6),
    (twos_complement, 5), (hex_convert, 4), (bit_ops, 6), (endianness, 5), (struct_layout, 6),
    (address_math, 3), (asm_trace, 8), (program_output, 8),
]
RE_TOPICS = {"twos_complement", "hex_to_dec", "dec_to_hex", "bit_ops", "little_endian", "little_endian_signed",
             "struct_sizeof", "struct_offset", "address", "asm_trace"}


def build(count: int, seed: int, he_share: float, samples: int) -> list[dict]:
    r = random.Random(seed)
    funcs = [f for f, w in GENERATORS for _ in range(w)]
    rows, seen = [], set()
    while len(rows) < count:
        he = r.random() < he_share
        made = r.choice(funcs)(r, he)
        if not made:
            continue
        topic, question, expected = made
        if question in seen:
            continue
        seen.add(question)
        rows.append(
            {
                "prompt": question + (ASK_HE if he else ASK_EN),
                "verify": {"kind": "verify_math", "expected": expected},
                "samples": samples,
                "max_tokens": 2048,
                "meta": {
                    "skill": "re_math" if topic in RE_TOPICS else ("code_trace" if topic == "program_output" else "math"),
                    "lang": "math", "hl": "he" if he else "en", "topic": topic, "source": SOURCE,
                    "id": f"math-v1-{seed}-{len(rows):05d}",
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hebrew", type=float, default=0.3, help="share of tasks written in Hebrew")
    parser.add_argument("--samples", type=int, default=4, help="answers per task (rejection sampling)")
    parser.add_argument("--out", default=str(HERE / "math-v1.jsonl"))
    args = parser.parse_args()
    rows = build(args.count, args.seed, args.hebrew, args.samples)
    # The answers must pass the same check the workers run, or the task is useless.
    sys.path.insert(0, str(HERE.parent))
    import mesh  # noqa: E402

    for row in rows:
        ok, verdict = mesh.job_verify_math({"answer": "so \\boxed{" + row["verify"]["expected"] + "}", "expected": row["verify"]["expected"]})
        if not (ok and verdict["passed"]):
            raise SystemExit(f"self-check failed for {row['meta']['id']}: {row['verify']['expected']}")
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    topics: dict[str, int] = {}
    for row in rows:
        topics[row["meta"]["topic"]] = topics.get(row["meta"]["topic"], 0) + 1
    print(f"wrote {len(rows)} tasks to {args.out}")
    print(", ".join(f"{k} {v}" for k, v in sorted(topics.items())))


if __name__ == "__main__":
    main()
