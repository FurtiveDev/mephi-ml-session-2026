import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib, os

df = pd.read_csv("data/UCI_Credit_Card.csv")
X = df.drop(columns=['ID', 'default.payment.next.month'])
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

model.fit(X_train, y_train)
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model_v1.pkl')
print(f"Accuracy: {model.score(X_test, y_test):.2f}")