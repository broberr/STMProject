# src/run_experiment.py
import os
import csv
import random
import re
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import f1_score

from src.config import Config
from src.frames import extract_frames_with_timestamps
from src.vlm_caption import VLMCaptioner
from src.llm_reason import ActivityClassifierLLM
from src.memory import ShortTermMemory
from src.utils_gpu import free_vram


ALLOWED_LABELS = ["meeting", "phone_use", "eating_drinking", "video_call", "using_computer"]


# -------------------------
# Helpers: label normalisation
# -------------------------
def norm_label(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return x.strip().lower().replace(" ", "_").replace("-", "_")


# -------------------------
# Dataset loader
# -------------------------
def load_dataset(videos_dir: str, labels_csv: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(labels_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video"].strip()
            lab = norm_label(row["label"])
            if not lab or lab not in ALLOWED_LABELS:
                continue
            path = os.path.join(videos_dir, vid)
            if os.path.exists(path):
                items.append((path, lab))
    return items


# -------------------------
# Observations builder (none vs stm)
# -------------------------
def build_observations(per_frame: List[Tuple[float, str]], mode: str, mem_cfg: dict) -> str:
    if mode == "none":
        return "\n".join([f"[{t:.1f}s] {cap}" for (t, cap) in per_frame])

    stm = ShortTermMemory(
        window_seconds=float(mem_cfg["window_seconds"]),
        compress=bool(mem_cfg["compress"]),
        summary_max_chars=int(mem_cfg["summary_max_chars"]),
    )
    for t, cap in per_frame:
        stm.add(t, cap)
    return stm.as_text()


# -------------------------
# Evidence scoring (closest-label fallback)
# -------------------------
def score_labels_from_text(text: str) -> Dict[str, int]:
    t = text.lower()

    patterns = {
        "video_call": [
            "video call", "on a call", "in a call", "zoom", "teams", "google meet", "meet",
            "facetime", "call controls", "webcam", "microphone", "headset",
            "speaking", "talking", "gesturing", "raised hand", "waving",
            "addressing the laptop", "engaged in a conversation",
        ],
        "phone_use": [
            "phone", "smartphone", "holding cell phone", "cell phone",
            "scrolling", "texting", "typing on phone", "holding phone",
            "calling", "answering calls", "answering", "pressing on screen",
        ],
        "eating_drinking": [
            "drinking", "sip", "sipping", "cup", "mug", "coffee", "tea", "eating", "bite",
        ],
        "meeting": [
            "meeting", "in-person", "with another person", "talking to someone", "two people",
        ],
        "using_computer": [
            "laptop", "computer", "typing", "touchpad", "keyboard",
            "attention towards computer", "looking at the laptop",
            "working on", "browsing", "reading", "writing",
            "using a laptop", "using the laptop",
        ],
    }

    # extra weight for very strong indicators
    strong_video_call = ["zoom", "teams", "google meet", "facetime", "call controls"]
    strong_phone_use = [
        "phone", "smartphone", "holding phone", "holding a phone", "cell phone",
        "scrolling", "texting", "typing on phone", "calling", "answering",
    ]

    raw = {k: 0 for k in patterns.keys()}

    for label, cues in patterns.items():
        for cue in cues:
            raw[label] += t.count(cue)

    for cue in strong_video_call:
        if cue in t:
            raw["video_call"] += 2

    for cue in strong_phone_use:
        if cue in t:
            raw["phone_use"] += 1

    # return only allowed labels (stable ordering)
    return {lab: int(raw.get(lab, 0)) for lab in ALLOWED_LABELS}


def pick_closest_label(scores: Dict[str, int]) -> str:
    scores = {k: v for k, v in scores.items() if k in ALLOWED_LABELS}
    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])

    # if absolutely no signal, choose safest generic
    if best_score == 0:
        return "using_computer"

    # tie-break: prefer more specific intents
    specificity_order = ["video_call", "phone_use", "eating_drinking", "meeting", "using_computer"]
    tied = [k for k, v in scores.items() if v == best_score]
    for lab in specificity_order:
        if lab in tied:
            return lab

    return best_label


# -------------------------
# LLM label prediction
# -------------------------
def predict_main_label_llm(classifier: ActivityClassifierLLM, observations_text: str) -> Optional[str]:
    prompt = f"""
You are a strict classifier.

Choose ONE label from this exact list:
{ALLOWED_LABELS}

IMPORTANT:
If you are not confident, output:
label=uncertain

Rules:
- Output EXACTLY one line: label=<one_label_or_uncertain>
- Use underscores, not spaces.
- Do NOT output explanations.

OBSERVATIONS:
{observations_text}
"""
    out = classifier.freeform(prompt)
    m = re.search(r"label\s*=\s*([a-zA-Z_ -]+)", out, flags=re.IGNORECASE)
    if not m:
        return None
    label = norm_label(m.group(1))
    if label in ALLOWED_LABELS or label == "uncertain":
        return label
    return None


def print_evidence(scores: Dict[str, int], closest: str):
    print("=== EVIDENCE SCORES ===\n")
    print(f"meeting         = {{{scores.get('meeting', 0)}}}")
    print(f"phone_use       = {{{scores.get('phone_use', 0)}}}")
    print(f"eating_drinking = {{{scores.get('eating_drinking', 0)}}}")
    print(f"video_call      = {{{scores.get('video_call', 0)}}}")
    print(f"using_computer  = {{{scores.get('using_computer', 0)}}}\n")
    print(f"closest={closest}\n")


def main():
    cfg = Config.load("config.yaml")

    # IMPORTANT: Config is a custom object, so use cfg.raw for optional keys
    seed = int(cfg.raw.get("seed", 42))
    random.seed(seed)

    data_cfg = cfg["data"]
    vlm_cfg = cfg["vlm"]
    llm_cfg = cfg["llm"]
    mem_cfg = cfg["memory"]
    exp_cfg = cfg["experiment"]

    items = load_dataset(data_cfg["videos_dir"], data_cfg["labels_csv"])
    if not items:
        print("No dataset items found. Check data.videos_dir and data.labels_csv paths.")
        return

    runs = int(exp_cfg.get("runs", 1))
    mode = mem_cfg.get("mode", "stm")

    f1s: List[float] = []

    for r in range(runs):
        random.shuffle(items)

        # ------------------------------------------------------------
        # Run-infer-like execution:
        #   For EACH video:
        #     VLM -> captions/observations -> free_vram()
        #     Evidence scores/closest
        #     LLM -> pred -> free_vram()
        #     Print per-video F1
        # ------------------------------------------------------------
        y_true: List[str] = []
        y_pred: List[str] = []

        for video_path, true_label in items:
            vid_name = os.path.basename(video_path)
            print(f"\n===== VIDEO: {vid_name} =====\n")

            # ---------- VLM phase (per video) ----------
            frames = extract_frames_with_timestamps(
                video_path,
                stride_seconds=float(data_cfg["frame_stride_seconds"]),
                max_frames=int(data_cfg["max_frames_per_video"]),
            )
            frames.sort(key=lambda x: x.timestamp_s)

            captioner = VLMCaptioner(
                model_id=vlm_cfg["model_id"],
                device=cfg["device"],
                dtype=cfg["dtype"],
                load_in_4bit=vlm_cfg["load_in_4bit"],
                max_new_tokens=vlm_cfg["max_new_tokens"],
            )

            per_frame: List[Tuple[float, str]] = []
            for fr in frames:
                cap = captioner.caption(fr.bgr, vlm_cfg["prompt"])
                cap = cap.replace("ASSISTANT:", "").replace("USER:", "").strip()
                per_frame.append((fr.timestamp_s, cap))

            observations = build_observations(per_frame, mode, mem_cfg)

            # FREE VLM before loading LLM (prevents VRAM crash / CPU offload)
            del captioner
            free_vram()

            # ---------- Evidence fallback ----------
            scores = score_labels_from_text(observations)
            closest = pick_closest_label(scores)
            print_evidence(scores, closest)

            # ---------- LLM phase (per video) ----------
            classifier = ActivityClassifierLLM(
                model_id=llm_cfg["model_id"],
                device=cfg["device"],
                dtype=cfg["dtype"],
                load_in_4bit=llm_cfg["load_in_4bit"],
                max_new_tokens=int(llm_cfg["max_new_tokens"]),
                temperature=float(llm_cfg["temperature"]),
                system_prompt=llm_cfg.get("system_prompt", ""),
            )

            llm_pred = predict_main_label_llm(classifier, observations)

            if llm_pred is None or llm_pred == "uncertain":
                pred_label = closest
            else:
                pred_label = llm_pred

            print("=== MAIN ACTION LABEL ===\n")
            print(f"pred={pred_label}")
            print(f"true={true_label}")

            per_video_f1 = f1_score([true_label], [pred_label], average="macro")
            print(f"F1_macro (single video) = {per_video_f1:.4f}")

            y_true.append(true_label)
            y_pred.append(pred_label)

            # FREE LLM
            del classifier
            free_vram()

        run_f1 = f1_score(y_true, y_pred, average="macro")
        f1s.append(run_f1)
        print(f"\nRun {r}: f1_macro={run_f1:.4f}\n")

    mean = sum(f1s) / len(f1s)
    var = sum((x - mean) ** 2 for x in f1s) / len(f1s)
    std = var ** 0.5
    print(f"\nAggregate: {{'runs': {runs}, 'f1_macro_mean': {mean:.4f}, 'f1_macro_std': {std:.4f}}}")


if __name__ == "__main__":
    main()