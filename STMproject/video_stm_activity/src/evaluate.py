from typing import List, Dict
from sklearn.metrics import f1_score, classification_report

def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    # macro F1 treats classes equally
    macro = f1_score(y_true, y_pred, average="macro")
    micro = f1_score(y_true, y_pred, average="micro")
    return {"f1_macro": macro, "f1_micro": micro}

def metrics_report(y_true: List[str], y_pred: List[str]) -> str:
    return classification_report(y_true, y_pred, digits=4)
