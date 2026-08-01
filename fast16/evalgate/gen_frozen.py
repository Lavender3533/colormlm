"""能力门冻结题库生成器。

题目参数随机生成、正确答案由代码计算,因此 ground truth 不依赖任何模型或人工判断。
固定种子保证题库可复现;题库一经冻结,禁止用于任何训练或调参。
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from fractions import Fraction
from math import comb, gcd
from pathlib import Path


def gen_divisibility(rng: random.Random, count: int) -> list[dict]:
    """1..N 中能被 a 或 b 整除但不能被 c 整除的个数。"""
    items = []
    seen = set()
    while len(items) < count:
        n = rng.randrange(200, 901, 10)
        a, b = sorted(rng.sample([3, 4, 6, 7, 8, 9, 11, 12, 13], 2))
        c = rng.choice([x for x in [2, 3, 4, 5, 6, 7, 9] if x not in (a, b)])
        key = (n, a, b, c)
        if key in seen:
            continue
        seen.add(key)
        answer = sum(
            1
            for x in range(1, n + 1)
            if (x % a == 0 or x % b == 0) and x % c != 0
        )
        items.append(
            {
                "family": "divisibility",
                "question": (
                    f"1 到 {n} 之间(含两端)能被 {a} 或 {b} 整除、"
                    f"但不能被 {c} 整除的整数有多少个?"
                ),
                "answer": str(answer),
                "answer_type": "int",
            }
        )
    return items


def gen_bayes(rng: random.Random, count: int) -> list[dict]:
    """经典阳性后验概率,答案为最简分数。"""
    items = []
    seen = set()
    while len(items) < count:
        p = rng.choice([20, 50, 100, 200, 500, 1000])
        s = rng.choice([80, 90, 95, 99])
        f = rng.choice([1, 2, 5, 10])
        key = (p, s, f)
        if key in seen:
            continue
        seen.add(key)
        posterior = Fraction(s, s + f * (p - 1))
        items.append(
            {
                "family": "bayes",
                "question": (
                    f"某疾病的患病率为 1/{p}。一种检测方法对患病者的阳性检出率为 {s}%,"
                    f"对未患病者的误报(假阳性)率为 {f}%。"
                    f"某人检测结果为阳性,那么这个人真正患病的概率是多少?"
                    f"请用最简分数作答。"
                ),
                "answer": f"{posterior.numerator}/{posterior.denominator}",
                "answer_type": "fraction",
            }
        )
    return items


_TT_TEMPLATES = [
    ("{x}是骗子", lambda types, x, y: not types[x]),
    ("{x}是诚实的", lambda types, x, y: types[x]),
    ("{x}和{y}都是骗子", lambda types, x, y: (not types[x]) and (not types[y])),
    ("{x}和{y}中恰好有一个骗子", lambda types, x, y: (types[x] != types[y])),
]


def gen_truthteller(rng: random.Random, count: int) -> list[dict]:
    """三人真假话,枚举 8 种指派,只保留唯一一致解的题。"""
    people = ["甲", "乙", "丙"]
    items = []
    seen = set()
    attempts = 0
    while len(items) < count and attempts < 20000:
        attempts += 1
        statements = []
        for i, speaker in enumerate(people):
            template, check = rng.choice(_TT_TEMPLATES)
            others = [p for p in people if p != speaker]
            x = rng.choice(others)
            y = [p for p in others if p != x][0]
            statements.append((speaker, template.format(x=x, y=y), check, x, y))
        key = tuple(s[1] for s in statements)
        if key in seen:
            continue
        consistent = []
        for assignment in itertools.product([True, False], repeat=3):
            types = dict(zip(people, assignment))
            ok = True
            for speaker, _text, check, x, y in statements:
                claim = check(types, x, y)
                if types[speaker] != claim:
                    ok = False
                    break
            if ok:
                consistent.append(types)
        if len(consistent) != 1:
            continue
        liars = sorted(p for p, honest in consistent[0].items() if not honest)
        if not liars:
            continue
        seen.add(key)
        lines = "".join(
            f"{speaker}说:「{text}」。" for speaker, text, _c, _x, _y in statements
        )
        items.append(
            {
                "family": "truthteller",
                "question": (
                    f"甲、乙、丙三人,每人要么永远说真话(诚实者),要么永远说假话(骗子)。"
                    f"{lines}"
                    f"骗子是谁?按顺序列出所有骗子,如「甲」或「甲,丙」。"
                ),
                "answer": ",".join(liars),
                "answer_type": "names",
            }
        )
    return items


def gen_ages(rng: random.Random, count: int) -> list[dict]:
    """今年 k 倍、d 年后 j 倍,解儿子年龄。由解反向构造,保证整数唯一。"""
    combos = []
    for k in [3, 4, 5, 6, 7]:
        for j in [2, 3]:
            if j >= k:
                continue
            for s in range(3, 31):
                d = s * (k - j)
                if d % (j - 1) != 0:
                    continue
                d //= j - 1
                if not 1 <= d <= 40:
                    continue
                combos.append((k, j, s, d))
    rng.shuffle(combos)
    items = []
    for k, j, s, d in combos[:count]:
        items.append(
            {
                "family": "ages",
                "question": (
                    f"今年,父亲的年龄是儿子的 {k} 倍;{d} 年后,"
                    f"父亲的年龄将是儿子的 {j} 倍。儿子今年多少岁?"
                ),
                "answer": str(s),
                "answer_type": "int",
            }
        )
    return items


def gen_remainder(rng: random.Random, count: int) -> list[dict]:
    """两模数同余,求最小正整数解,暴力枚举到 lcm 保证正确。"""
    items = []
    seen = set()
    while len(items) < count:
        m1, m2 = sorted(rng.sample([5, 6, 7, 8, 9, 11, 12, 13], 2))
        if gcd(m1, m2) != 1:
            continue
        r1 = rng.randrange(1, m1)
        r2 = rng.randrange(1, m2)
        key = (m1, r1, m2, r2)
        if key in seen:
            continue
        seen.add(key)
        lcm = m1 * m2
        answer = next(
            x for x in range(1, lcm + 1) if x % m1 == r1 and x % m2 == r2
        )
        items.append(
            {
                "family": "remainder",
                "question": (
                    f"一个正整数除以 {m1} 余 {r1},除以 {m2} 余 {r2}。"
                    f"满足条件的最小正整数是多少?"
                ),
                "answer": str(answer),
                "answer_type": "int",
            }
        )
    return items


def gen_combinatorics(rng: random.Random, count: int) -> list[dict]:
    """带约束的组合选取,comb() 直接计算。"""
    items = []
    seen = set()
    while len(items) < count:
        boys = rng.randrange(4, 9)
        girls = rng.randrange(3, 8)
        k = rng.randrange(3, min(boys + girls, 7))
        min_girls = rng.randrange(1, min(girls, k) + 1)
        key = (boys, girls, k, min_girls)
        if key in seen:
            continue
        seen.add(key)
        answer = sum(
            comb(girls, g) * comb(boys, k - g)
            for g in range(min_girls, min(girls, k) + 1)
            if 0 <= k - g <= boys
        )
        if answer <= 1:
            continue
        items.append(
            {
                "family": "combinatorics",
                "question": (
                    f"从 {boys} 名男生和 {girls} 名女生中选出 {k} 人组成小组,"
                    f"要求其中至少有 {min_girls} 名女生,共有多少种不同的选法?"
                ),
                "answer": str(answer),
                "answer_type": "int",
            }
        )
    return items


GENERATORS = [
    gen_divisibility,
    gen_bayes,
    gen_truthteller,
    gen_ages,
    gen_remainder,
    gen_combinatorics,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成冻结能力门题库")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "frozen_v1.json",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items: list[dict] = []
    for generator in GENERATORS:
        items.extend(generator(rng, args.per_family))
    for index, item in enumerate(items):
        item["id"] = f"q{index:03d}"

    payload = {
        "seed": args.seed,
        "per_family": args.per_family,
        "total": len(items),
        "frozen": True,
        "note": "冻结题库:禁止用于训练、调参或提示词迭代;只用于前后对照评测。",
        "items": items,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"已生成 {len(items)} 题 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
