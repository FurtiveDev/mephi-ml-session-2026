import pytest, requests

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_models():
    r = requests.get(f"{BASE_URL}/models")
    assert "v1" in r.json()["models"]

def test_predict_v1():
    data = {"LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 50000, "BILL_AMT2": 50000, "BILL_AMT3": 50000, "BILL_AMT4": 50000, "BILL_AMT5": 50000, "BILL_AMT6": 50000, "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0, "model_version": "v1"}
    r = requests.post(f"{BASE_URL}/predict", json=data)
    assert r.status_code == 200
    assert r.json()["version"] == "v1"

def test_predict_v2():
    data = {"LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 50000, "BILL_AMT2": 50000, "BILL_AMT3": 50000, "BILL_AMT4": 50000, "BILL_AMT5": 50000, "BILL_AMT6": 50000, "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0, "model_version": "v2"}
    r = requests.post(f"{BASE_URL}/predict", json=data)
    assert r.status_code == 200
    assert r.json()["version"] == "v2"

def test_predict_ab():
    data = {"user_id": "c1", "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 50000, "BILL_AMT2": 50000, "BILL_AMT3": 50000, "BILL_AMT4": 50000, "BILL_AMT5": 50000, "BILL_AMT6": 50000, "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}
    r = requests.post(f"{BASE_URL}/predict/ab", json=data)
    assert r.status_code == 200

def test_ab_no_user():
    r = requests.post(f"{BASE_URL}/predict/ab", json={"LIMIT_BAL": 50000})
    assert r.status_code == 400

def test_missing():
    r = requests.post(f"{BASE_URL}/predict", json={"LIMIT_BAL": 50000})
    assert r.status_code == 400