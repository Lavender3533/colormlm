"""Deterministic BM25 bridge from compiled CLM records to the GGUF core."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ASCII_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_.+\-]{1,31}")
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
CJK_STOP = {
    "一个",
    "写一",
    "什么",
    "这是",
    "怎么",
    "如何",
    "是否",
    "可以",
    "关系",
    "一下",
    "的吗",
    "什么",
}


def _terms(text: str) -> list[str]:
    terms = [term.lower() for term in ASCII_TERM.findall(text)]
    for run in CJK_RUN.findall(text):
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(
                term
                for index in range(len(run) - 1)
                if (term := run[index : index + 2]) not in CJK_STOP
            )
    return terms


@dataclass(frozen=True)
class MemoryHit:
    key: str
    value: str
    score: float


class CompiledMemoryRetriever:
    def __init__(self, paths: list[str | Path]):
        self.records: list[dict[str, str]] = []
        for raw_path in paths:
            with Path(raw_path).open("r", encoding="utf-8") as source:
                self.records.extend(json.loads(line) for line in source if line.strip())

        self.key_counts: list[Counter[str]] = []
        self.body_counts: list[Counter[str]] = []
        self.lengths: list[int] = []
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for record in self.records:
            key_counts = Counter(_terms(record["key"]))
            body_counts = Counter(_terms(record["value"]))
            counts = key_counts + body_counts
            self.key_counts.append(key_counts)
            self.body_counts.append(body_counts)
            length = sum(counts.values())
            self.lengths.append(length)
            for term in counts:
                document_frequency[term] += 1

        count = max(len(self.records), 1)
        self.average_length = sum(self.lengths) / count
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str) -> MemoryHit | None:
        query_terms = Counter(_terms(query))
        if not query_terms or not self.records:
            return None
        best_index = -1
        best_score = 0.0
        k1 = 1.2
        b = 0.75
        for index, (key_counts, body_counts) in enumerate(zip(self.key_counts, self.body_counts)):
            norm = k1 * (1.0 - b + b * self.lengths[index] / max(self.average_length, 1.0))
            key_score = 0.0
            body_score = 0.0
            for term, query_count in query_terms.items():
                idf = self.idf.get(term, 0.0)
                key_frequency = key_counts.get(term, 0)
                body_frequency = body_counts.get(term, 0)
                if key_frequency:
                    key_score += idf * (key_frequency * (k1 + 1.0) / (key_frequency + norm)) * query_count
                if body_frequency:
                    body_score += idf * (body_frequency * (k1 + 1.0) / (body_frequency + norm)) * query_count
            if key_score == 0.0:
                continue
            score = 3.0 * key_score + 0.35 * body_score
            if score > best_score:
                best_index = index
                best_score = score
        if best_index < 0:
            return None
        record = self.records[best_index]
        return MemoryHit(record["key"], record["value"], best_score)
