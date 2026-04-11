from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'raw_data_analysis' / 'master_table_v1.csv'
OUT_DIR = BASE_DIR / 'benchmark_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TOP_K = 3


def build_text(df: pd.DataFrame) -> pd.Series:
    tag1 = df['message_tag_level1'].fillna('').astype(str)
    tag2 = df['message_tag_level2'].fillna('').astype(str)
    title = df['message_title'].fillna('').astype(str)
    body = df['message_body'].fillna('').astype(str)
    district = df['district_from_file'].fillna('').astype(str)
    return ('【' + tag1 + '】【' + tag2 + '】【' + district + '】' + title + '。' + body).str.replace(r'\s+', ' ', regex=True).str.strip()


def load_dataset() -> tuple[pd.Series, np.ndarray, dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'未找到母表文件: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig', low_memory=False)
    mask = df['recommended_for_unit_cls'].astype(str).str.lower().isin(['true', '1'])
    sub = df.loc[mask].copy()
    sub = sub[sub['message_title'].fillna('').ne('') | sub['message_body'].fillna('').ne('')]
    sub = sub[sub['reply_unit_norm'].fillna('').ne('')]
    texts = build_text(sub)
    labels = sub['reply_unit_norm'].astype(str).str.strip()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    meta = {
        'rows': int(len(sub)),
        'classes': int(len(le.classes_)),
        'class_lt_2': int(pd.Series(y).value_counts().lt(2).sum()),
        'class_lt_5': int(pd.Series(y).value_counts().lt(5).sum()),
    }
    return texts, y, meta


def topk_accuracy_from_scores(scores: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    k = min(k, scores.shape[1])
    topk = np.argpartition(scores, -k, axis=1)[:, -k:]
    hits = (topk == y_true[:, None]).any(axis=1)
    return float(hits.mean())


def scores_from_estimator(model, X):
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    raise RuntimeError(f'Estimator {model.__class__.__name__} has neither decision_function nor predict_proba')


def fit_vectorizer(train_texts: pd.Series, val_texts: pd.Series):
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=30000,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_texts)
    X_val = vec.transform(val_texts)
    return vec, X_train, X_val


def build_models():
    models = []
    models.append({
        'name': 'LinearSVC + TF-IDF',
        'family': 'linear_margin',
        'use_svd': False,
        'estimator': LinearSVC(C=1.0, class_weight='balanced', random_state=RANDOM_STATE),
    })
    models.append({
        'name': 'LogisticRegression + TF-IDF',
        'family': 'linear_probabilistic',
        'use_svd': False,
        'estimator': LogisticRegression(
            max_iter=1000,
            solver='saga',
            n_jobs=-1,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            multi_class='auto',
        ),
    })
    models.append({
        'name': 'ComplementNB + TF-IDF',
        'family': 'naive_bayes',
        'use_svd': False,
        'estimator': ComplementNB(alpha=0.5),
    })
    models.append({
        'name': 'SGDClassifier(log_loss) + TF-IDF',
        'family': 'linear_online',
        'use_svd': False,
        'estimator': SGDClassifier(
            loss='log_loss',
            alpha=1e-5,
            max_iter=2000,
            tol=1e-3,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    })
    models.append({
        'name': 'PassiveAggressive + TF-IDF',
        'family': 'online_margin',
        'use_svd': False,
        'estimator': PassiveAggressiveClassifier(
            C=0.5,
            max_iter=2000,
            tol=1e-3,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        ),
    })
    models.append({
        'name': 'RandomForest + TF-IDF-SVD',
        'family': 'bagging_tree',
        'use_svd': True,
        'svd_dim': 300,
        'estimator': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight='balanced_subsample',
        ),
    })
    models.append({
        'name': 'ExtraTrees + TF-IDF-SVD',
        'family': 'extremely_randomized_tree',
        'use_svd': True,
        'svd_dim': 300,
        'estimator': ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight='balanced',
        ),
    })
    return models


def run():
    texts, y, meta = load_dataset()
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )

    vec_start = time.perf_counter()
    vectorizer, X_train_tfidf, X_val_tfidf = fit_vectorizer(X_train_text, X_val_text)
    vec_seconds = time.perf_counter() - vec_start

    results = []
    models = build_models()
    for cfg in models:
        model_name = cfg['name']
        print(f'=== Running: {model_name} ===', flush=True)
        fit_X_train = X_train_tfidf
        fit_X_val = X_val_tfidf
        preprocess_note = 'tfidf'
        svd_seconds = 0.0
        if cfg.get('use_svd'):
            svd_start = time.perf_counter()
            svd = TruncatedSVD(n_components=cfg['svd_dim'], random_state=RANDOM_STATE)
            normalizer = Normalizer(copy=False)
            fit_X_train = normalizer.fit_transform(svd.fit_transform(X_train_tfidf))
            fit_X_val = normalizer.transform(svd.transform(X_val_tfidf))
            svd_seconds = time.perf_counter() - svd_start
            preprocess_note = f'tfidf+svd{cfg["svd_dim"]}'

        estimator = clone(cfg['estimator'])
        train_start = time.perf_counter()
        estimator.fit(fit_X_train, y_train)
        train_seconds = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        y_pred = estimator.predict(fit_X_val)
        scores = scores_from_estimator(estimator, fit_X_val)
        infer_seconds = time.perf_counter() - infer_start

        top1 = accuracy_score(y_val, y_pred)
        top3 = topk_accuracy_from_scores(np.asarray(scores), y_val, TOP_K)
        macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        row = {
            'model': model_name,
            'family': cfg['family'],
            'preprocess': preprocess_note,
            'train_seconds': round(train_seconds, 4),
            'preprocess_seconds': round(svd_seconds, 4),
            'vectorizer_seconds': round(vec_seconds, 4),
            'infer_seconds_total': round(infer_seconds, 4),
            'infer_ms_per_sample': round(infer_seconds * 1000 / len(y_val), 4),
            'val_rows': int(len(y_val)),
            'top1_accuracy': round(float(top1), 6),
            'top3_accuracy': round(float(top3), 6),
            'macro_f1': round(float(macro_f1), 6),
            'weighted_f1': round(float(weighted_f1), 6),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    result_df = pd.DataFrame(results).sort_values(['top1_accuracy', 'top3_accuracy', 'macro_f1'], ascending=False)
    result_df.to_csv(OUT_DIR / 'phase1_benchmark_results.csv', index=False, encoding='utf-8-sig')

    report = {
        'dataset_meta': meta,
        'train_rows': int(len(y_train)),
        'val_rows': int(len(y_val)),
        'vectorizer': {
            'max_features': 30000,
            'ngram_range': [1, 2],
            'min_df': 2,
            'max_df': 0.98,
            'vectorizer_seconds': round(vec_seconds, 4),
        },
        'results_sorted': result_df.to_dict(orient='records'),
    }
    (OUT_DIR / 'phase1_benchmark_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('=== DONE ===')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
