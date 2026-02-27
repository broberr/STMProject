from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class MemoryItem:
    t: float
    caption: str

class ShortTermMemory:
    def __init__(self, window_seconds: float, compress: bool, summary_max_chars: int):
        self.window_seconds = window_seconds
        self.compress = compress
        self.summary_max_chars = summary_max_chars

        self.items: List[MemoryItem] = []
        self.summary: str = ""  # rolling compression

    def add(self, t: float, caption: str):
        self.items.append(MemoryItem(t=t, caption=caption))
        self._trim(t)

        if self.compress:
            # lightweight rolling “summary” (no LLM call).
            # You can later replace with LLM-based summariser if you want.
            line = f"[{t:.1f}s] {caption}"
            if not self.summary:
                self.summary = line
            else:
                self.summary = self.summary + " | " + line

            # hard cap
            if len(self.summary) > self.summary_max_chars:
                self.summary = self.summary[-self.summary_max_chars:]

    def _trim(self, current_t: float):
        cutoff = current_t - self.window_seconds
        self.items = [x for x in self.items if x.t >= cutoff]

    def as_text(self) -> str:
        if self.compress and self.summary:
            return f"MEMORY_SUMMARY: {self.summary}"
        # raw (more tokens)
        return "\n".join([f"[{x.t:.1f}s] {x.caption}" for x in self.items])
