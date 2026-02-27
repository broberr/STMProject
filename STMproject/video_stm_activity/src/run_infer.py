# src/run_infer.py old
import argparse
import csv
import re

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
def norm_label(x: str) -> str:
    if x is None:
        return None
    return x.strip().lower().replace(" ", "_").replace("-", "_")


# -------------------------
# Observations builder (none vs stm)
# -------------------------
def build_observations(per_frame, mode: str) -> str | None:
    """
    mode:
      - "none": return raw timestamped captions (no STM logic at all)
      - "stm": return None, caller will build via ShortTermMemory
    """
    if mode == "none":
        lines = [f"[{t:.1f}s] {cap}" for (t, cap) in per_frame]
        return "\n".join(lines)
    return None


# -------------------------
# Evidence scoring (closest-label fallback)
# -------------------------
def score_labels_from_text(text: str) -> dict:
    t = text.lower()

    patterns = {
        "video_call": [
            "video call", "on a call", "in a call", "zoom", "teams", "google meet", "meet",
            "facetime", "call controls", "webcam", "microphone", "headset",
            "speaking", "talking", "gesturing", "raised hand", "waving",
            "addressing the laptop", "engaged in a conversation"
        ],
        "phone_use": [
            "phone", "smartphone", "scrolling", "texting","holding cell phone", "typing on phone", "holding phone", "calling" ,
            "answering calls", "pressing on screen"
        ],
        "eating_drinking": [
            "drinking", "sip", "sipping", "cup", "mug", "coffee", "tea", "eating", "bite"
        ],
        "meeting": [
            "meeting", "in-person", "with another person", "talking to someone", "two people"
        ],
        "using_computer": [
            "laptop", "computer", "typing", "touchpad", "keyboard","attention towards computer","looking at the laptop",
            "working on", "browsing", "reading", "writing", "using a laptop", "using the laptop"
        ],
    }

    # extra weight for very strong video-call indicators
    strong_video_call = ["zoom", "teams", "google meet", "facetime", "call controls"]

    strong_phone_use = [
        "phone", "smartphone", "holding phone", "holding a phone", "cell phone",
        "scrolling", "texting", "typing on phone", "calling", "answering"
    ]
    counts = {k: 0 for k in patterns.keys()}

    for label, cues in patterns.items():
        for cue in cues:
            counts[label] += t.count(cue)

    for cue in strong_video_call:
        if cue in t:
            counts["video_call"] += 2

    for cue in strong_phone_use:
        if cue in t:
            counts["phone_use"] += 1

    return {lab: int(counts.get(lab, 0)) for lab in ALLOWED_LABELS}


def pick_closest_label(scores: dict) -> str:
    # keep allowed only
    scores = {k: v for k, v in scores.items() if k in ALLOWED_LABELS}

    best_label, best_score = max(scores.items(), key=lambda kv: kv[1])

    # if absolutely no signal, choose safest generic label
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
# Labels CSV loader
# -------------------------
def load_video_labels(csv_path: str) -> dict:
    labels = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["video"].strip()] = norm_label(row["label"])
    return labels


# -------------------------
# LLM label prediction
# -------------------------
def predict_main_label_llm(classifier, observations_text: str) -> str | None:
    prompt = f"""
You are a strict classifier.

Choose ONE label from this exact list:
{ALLOWED_LABELS}

Rules:
- Output EXACTLY one line: label=<one_label>
- Use underscores, not spaces (e.g., using_computer).
- Do NOT output explanations.

OBSERVATIONS:
{observations_text}
"""
    out = classifier.freeform(prompt)

    # parse label=<...>
    m = re.search(r"label\s*=\s*([a-zA-Z_ -]+)", out, flags=re.IGNORECASE)
    if not m:
        return None

    label = norm_label(m.group(1))
    if label not in ALLOWED_LABELS and label != "uncertain":
        return None
    return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    cfg = Config.load("config.yaml")
    data_cfg = cfg["data"]
    vlm_cfg = cfg["vlm"]
    llm_cfg = cfg["llm"]
    mem_cfg = cfg["memory"]

    # Ground truth
    labels_map = load_video_labels(data_cfg["labels_csv"])
    video_name = args.video.split("\\")[-1].split("/")[-1]
    true_label = labels_map.get(video_name, None)

    # Frames
    frames = extract_frames_with_timestamps(
        args.video,
        stride_seconds=float(data_cfg["frame_stride_seconds"]),
        max_frames=int(data_cfg["max_frames_per_video"]),
    )
    frames.sort(key=lambda x: x.timestamp_s)

    # -------------------------
    # 1) VLM captions
    # -------------------------
    captioner = VLMCaptioner(
        model_id=vlm_cfg["model_id"],
        device=cfg["device"],
        dtype=cfg["dtype"],
        load_in_4bit=vlm_cfg["load_in_4bit"],
        max_new_tokens=vlm_cfg["max_new_tokens"],
    )

    per_frame = []
    for fr in frames:
        cap = captioner.caption(fr.bgr, vlm_cfg["prompt"])
        cap = cap.replace("ASSISTANT:", "").replace("USER:", "").strip()
        per_frame.append((fr.timestamp_s, cap))

    print("\n=== PER-FRAME CAPTIONS ===\n")
    for t, cap in per_frame:
        print(f"[{t:06.2f}s] {cap}")

    # -------------------------
    # 2) Build OBSERVATIONS text (none vs stm)
    # -------------------------
    mode = mem_cfg.get("mode", "stm")
    observations_text = build_observations(per_frame, mode)

    if observations_text is None:
        stm = ShortTermMemory(
            window_seconds=float(mem_cfg["window_seconds"]),
            compress=bool(mem_cfg["compress"]),
            summary_max_chars=int(mem_cfg["summary_max_chars"]),
        )
        for t, cap in per_frame:
            stm.add(t, cap)
        observations_text = stm.as_text()

    print("\n=== OBSERVATIONS ===\n")
    print(observations_text[:4000], "...\n" if len(observations_text) > 4000 else "\n")

    # Evidence scores always computed from the same observations used for the LLM
    scores = score_labels_from_text(observations_text)
    closest = pick_closest_label(scores)

    print("\n=== EVIDENCE SCORES ===\n")
    print(scores)
    print(f"\nclosest={closest}")

    # Free VLM before LLM
    del captioner
    free_vram()

    # -------------------------
    # 3) LLM classification (but safe fallback)
    # -------------------------
    classifier = ActivityClassifierLLM(
        model_id=llm_cfg["model_id"],
        device=cfg["device"],
        dtype=cfg["dtype"],
        load_in_4bit=llm_cfg["load_in_4bit"],
        max_new_tokens=int(llm_cfg["max_new_tokens"]),
        temperature=float(llm_cfg["temperature"]),
        system_prompt=llm_cfg.get("system_prompt", ""),
    )

    llm_pred = predict_main_label_llm(classifier, observations_text)

    # If LLM fails / outputs invalid label -> fallback to closest evidence label
    if llm_pred is None or llm_pred == "uncertain":
        pred_label = closest
    else:
        pred_label = llm_pred

    print("\n=== MAIN ACTION LABEL ===\n")
    print(f"pred={pred_label}")

    if true_label is None:
        print("true=<missing label in labels csv>")
    else:
        print(f"true={true_label}")
        f1 = f1_score([true_label], [pred_label], average="macro")
        print(f"F1_macro (single video) = {f1:.4f}")

    # free LLM
    del classifier
    free_vram()


if __name__ == "__main__":
    main()