import numpy as np
from sklearn.linear_model import LogisticRegression
from src.experiment import evade, data

def test_constraints_and_query_budget():
    Xtr, Xte, ytr, _ = data(n_samples=200)
    model=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    changed, queries=evade(model,Xte[:12],budget=12,seed=3)
    assert changed.shape == Xte[:12].shape
    assert np.array_equal(changed[:,3:],Xte[:12,3:])
    assert np.all(np.abs(changed[:,:3]-Xte[:12,:3])<=0.8000001)
    assert np.all(queries<=12)

def test_deterministic():
    Xtr, Xte, ytr, _=data(n_samples=200)
    model=LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    a,_=evade(model,Xte[:10],budget=10,seed=4)
    b,_=evade(model,Xte[:10],budget=10,seed=4)
    assert np.array_equal(a,b)
