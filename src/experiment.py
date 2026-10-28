"""Reproducible educational black-box evasion study on synthetic tabular traffic.

Not a real-world intrusion-detection benchmark. No packet manipulation or network activity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MUTABLE = (0, 1, 2)

def data(seed=42, n_samples=1400):
    X, y = make_classification(n_samples=n_samples, n_features=12, n_informative=8,
        n_redundant=2, weights=[0.65, 0.35], random_state=seed)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

def models(seed):
    return {
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        'random_forest': RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(random_state=seed),
        'mlp': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(24,), max_iter=350, early_stopping=True, random_state=seed)),
    }

def evade(model, X, *, budget=150, seed=42, mutable=MUTABLE, max_delta=0.8):
    """Query-limited black-box search, constrained to allowed feature indices."""
    rng = np.random.default_rng(seed)
    original = np.asarray(X, dtype=float)
    result = original.copy()
    queries = np.zeros(len(result), dtype=int)
    for i, x in enumerate(original):
        if budget < 1:
            continue
        queries[i] += 1
        if model.predict(x.reshape(1,-1))[0] != 1:
            continue
        for _ in range(budget - 1):
            candidate = x.copy()
            candidate[list(mutable)] += rng.uniform(-max_delta, max_delta, len(mutable))
            queries[i] += 1
            if model.predict(candidate.reshape(1,-1))[0] == 0:
                result[i] = candidate
                break
    return result, queries

def run(seed=42, budget=150):
    X_train, X_test, y_train, y_test = data(seed)
    outcomes = {}
    for name, model in models(seed).items():
        model.fit(X_train, y_train)
        positive = X_test[y_test == 1]
        initial = model.predict(positive) == 1
        attacked, queries = evade(model, positive, budget=budget, seed=seed)
        after = model.predict(attacked)
        successes = int(np.sum(initial & (after == 0)))
        outcomes[name] = {
            'clean_accuracy': round(float(accuracy_score(y_test, model.predict(X_test))), 4),
            'clean_recall': round(float(recall_score(y_test, model.predict(X_test))), 4),
            'initially_detected_attacks': int(initial.sum()),
            'successful_evasions': successes,
            'evasion_rate_among_initially_detected': round(successes / max(1,int(initial.sum())), 4),
            'mean_queries_per_attack_sample': round(float(queries.mean()), 2),
        }
    return {'dataset': 'synthetic sklearn classification (NOT NSL-KDD)', 'seed':seed, 'query_budget':budget, 'results':outcomes}

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--budget',type=int,default=150)
    parser.add_argument('--output',default='results/results.json')
    args=parser.parse_args()
    result=run(args.seed,args.budget)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
