from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging

app = Flask(__name__)
models = {}

is_docker = os.path.exists('/app')
MODEL_DIR = '/app/models' if is_docker else 'models'

model_paths = {'v1': f'{MODEL_DIR}/model_v1.pkl', 'v2': f'{MODEL_DIR}/model_v2.pkl'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

default_features = {"LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 5000, "BILL_AMT2": 4500, "BILL_AMT3": 4000, "BILL_AMT4": 3500, "BILL_AMT5": 3000, "BILL_AMT6": 2500, "PAY_AMT1": 1000, "PAY_AMT2": 900, "PAY_AMT3": 800, "PAY_AMT4": 700, "PAY_AMT5": 600, "PAY_AMT6": 500}

def load_models():
    global models
    for version, path in model_paths.items():
        models[version] = joblib.load(path)
        logger.info(f"Model {version} loaded")

def get_ab_group(user_id):
    return 'v1' if hash(str(user_id)) % 100 < 50 else 'v2'

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models": list(models.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    version = data.pop('model_version', None) or get_ab_group(data.get('user_id'))
    
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    
    input_df = pd.DataFrame([{f: data.get(f, 0) for f in FEATURES}])
    prob = models[version].predict_proba(input_df)[0][1]
    pred = models[version].predict(input_df)[0]
    
    return jsonify({"prediction": int(pred), "prob": round(float(prob), 4), "version": version})

@app.route("/predict/ab", methods=["POST"])
def predict_ab():
    data = request.get_json() or {}
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Missing: {missing}"}), 400
    
    version = get_ab_group(user_id)
    input_df = pd.DataFrame([{f: data.get(f, 0) for f in FEATURES}])
    prob = models[version].predict_proba(input_df)[0][1]
    pred = models[version].predict(input_df)[0]
    
    return jsonify({"prediction": int(pred), "prob": round(float(prob), 4), "version": version, "group": version})

@app.route("/models")
def list_models():
    return jsonify({"models": list(models.keys())})

if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server started on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)