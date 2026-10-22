"""Testnivå 3: modell. Kontrakt och reproducerbarhet."""

import json
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
import pytest

from churn.data import FEATURES, dela_upp, las_data, skapa_features
from churn.model import trana_och_utvardera


@pytest.fixture(scope="module")
def resultat():
    return trana_och_utvardera()


def test_modellkontrakt(resultat):
    modell, _ = resultat
    X, _ = dela_upp(skapa_features(las_data().head(5)))
    assert list(X.columns) == FEATURES
    sannolikhet = modell.predict_proba(X)
    assert sannolikhet.shape == (5, 2)
    assert np.all((sannolikhet >= 0) & (sannolikhet <= 1))


def test_reproducerbar(resultat):
    _, matvarden = resultat
    _, igen = trana_och_utvardera()
    assert igen == matvarden



def test_senaste_traningen_ar_godkand():
    # Lades till efter förra incidenten: kolla att senaste träningen har ett rimligt ROC AUC.
    matvarden = json.loads(Path("outputs/matvarden.json").read_text())
    assert matvarden["roc_auc"] >= 0.70

def test_dummy_score(resultat):
    _, matvarden = resultat
    assert matvarden["roc_auc"] >= 0.80

    X, y = dela_upp(skapa_features(las_data()))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    dummy_mod = DummyClassifier(strategy="prior")
    dummy_mod.fit(X_train, y_train)
    dummy_score = roc_auc_score(y_test, dummy_mod.predict_proba(X_test)[:, 1])
    assert matvarden["roc_auc"] > dummy_score
