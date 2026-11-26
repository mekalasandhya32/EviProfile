"""Stage 5-7: resume classification (TF-IDF baselines, MLP deep model, evidence-fused MLP)."""
from __future__ import annotations
import re
import time
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from .lexicons import CAT, CATEGORIES
from .extractor import default_matcher
from .preprocess import model_text

TOKEN_PATTERN = r"[a-z][a-z0-9+#.\-]*[a-z0-9+#]|[a-z]"


def make_tfidf(max_features: int = 8000) -> TfidfVectorizer:
    return TfidfVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           max_features=max_features, lowercase=True)


class KeywordRule(BaseEstimator, ClassifierMixin):
    """Baseline that mimics a classic ATS: count category keywords anywhere in the text, take the arg-max."""

    def fit(self, X, y=None):
        self.classes_ = np.array(CATEGORIES)
        return self

    def _score(self, text: str):
        m = default_matcher()
        found = {s for s, _, _ in m.find(text)}
        return np.array([sum(1.0 for s in found if s in CAT[c]["core"]) + sum(0.5 for s in found if s in CAT[c]["secondary"])
                         for c in CATEGORIES])

    def predict(self, X):
        return np.array([CATEGORIES[int(np.argmax(self._score(t)))] for t in X])

    def predict_proba(self, X):
        S = np.array([self._score(t) for t in X]) + 1e-6
        return S / S.sum(1, keepdims=True)


def structured_features(parsed: dict, ev: dict, cred: dict) -> np.ndarray:
    """26-dim evidence feature vector: per-category SES affinity, per-category title match, profile scalars."""
    aff = []
    for c in CATEGORIES:
        d = CAT[c]
        tot = sum(ev[s]["ses"] * (1.0 if s in d["core"] else 0.5) for s in ev if s in d["core"] or s in d["secondary"])
        aff.append(min(1.0, tot / 6.0))
    title = (parsed.get("recent_title") or "").lower()
    tmatch = [1.0 if any(title == t.lower() for t in CAT[c]["titles"]) else 0.0 for c in CATEGORIES]
    top = sorted((v["ses"] for v in ev.values()), reverse=True)[:8]
    scal = [min(1.0, parsed["dated_years"] / 15.0), parsed["education"]["level"] / 4.0,
            min(1.0, len(parsed["narrative_skills"]) / 20.0), min(1.0, parsed["n_listed"] / 25.0), cred["pci"],
            float(np.mean(top)) if top else 0.0]
    return np.array(aff + tmatch + scal, dtype=float)


def make_mlp(seed: int = 0, hidden=(256, 64)) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam", alpha=1e-3, batch_size=64,
                         learning_rate_init=1e-3, max_iter=80, early_stopping=True, validation_fraction=0.12,
                         n_iter_no_change=6, random_state=seed)


def baseline_models(seed: int = 0) -> Dict[str, object]:
    return {
        "Multinomial NB": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=300, C=10.0),
        "Linear SVM": LinearSVC(C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=120, n_jobs=1, random_state=seed),
        "MLP (TF-IDF)": make_mlp(seed),
    }


class FusedMLP(BaseEstimator, ClassifierMixin):
    """MLP over [TF-IDF(blind text) | w * evidence features]."""

    def __init__(self, seed: int = 0, w: float = 2.0, hidden=(256, 64)):
        self.seed, self.w, self.hidden = seed, w, hidden

    def fit(self, X_text, X_struct, y):
        self.mlp_ = make_mlp(self.seed, self.hidden)
        self.mlp_.fit(sp.hstack([X_text, sp.csr_matrix(X_struct * self.w)]).tocsr(), y)
        self.classes_ = self.mlp_.classes_
        return self

    def _x(self, X_text, X_struct):
        return sp.hstack([X_text, sp.csr_matrix(X_struct * self.w)]).tocsr()

    def predict(self, X_text, X_struct):
        return self.mlp_.predict(self._x(X_text, X_struct))

    def predict_proba(self, X_text, X_struct):
        return self.mlp_.predict_proba(self._x(X_text, X_struct))


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t
