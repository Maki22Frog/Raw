from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .decision_support import apply_decision_support


def load_bundle(path: str | Path) -> Dict[str, object]:
    import joblib

    return joblib.load(path)


def predict_feature_frame(bundle: Dict[str, object], features: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in bundle["feature_columns"] if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing[:10]}")

    X = features[bundle["feature_columns"]]
    X_np = bundle["imputer"].transform(X)
    selected_indices = bundle.get("selected_feature_indices")
    if selected_indices is not None:
        X_np = X_np[:, np.asarray(selected_indices, dtype=int)]
    scaler = bundle.get("scaler")
    if scaler is not None:
        X_np = scaler.transform(X_np)

    model = bundle["model"]
    encoder = bundle["label_encoder"]
    proba: Optional[np.ndarray] = model.predict_proba(X_np) if hasattr(model, "predict_proba") else None
    threshold = bundle.get("decision_threshold")
    if threshold is not None and proba is not None and proba.shape[1] == 2:
        classes = np.asarray(model.classes_, dtype=int)
        positive_encoded = 1 if 1 in classes else int(classes[-1])
        pos_idx = int(np.flatnonzero(classes == positive_encoded)[0])
        neg_encoded = int([cls for cls in classes.tolist() if cls != positive_encoded][0])
        pred_encoded = np.where(proba[:, pos_idx] >= float(threshold), positive_encoded, neg_encoded)
    else:
        pred_encoded = model.predict(X_np)
    pred = encoder.inverse_transform(pred_encoded.astype(int))

    out = features.copy()
    out["pred_label"] = pred
    if proba is not None:
        class_to_idx = {int(label): idx for idx, label in enumerate(model.classes_.tolist())}
        confidence = np.full(len(pred), np.nan)
        for row_idx, encoded_label in enumerate(pred_encoded.astype(int)):
            class_idx = class_to_idx.get(int(encoded_label))
            if class_idx is not None and class_idx < proba.shape[1]:
                confidence[row_idx] = proba[row_idx, class_idx]
        out["confidence"] = confidence
        for idx, class_label in enumerate(encoder.classes_):
            if idx < proba.shape[1]:
                out[f"proba_{int(class_label)}"] = proba[:, idx]
    else:
        out["confidence"] = np.nan
    return apply_decision_support(out)
