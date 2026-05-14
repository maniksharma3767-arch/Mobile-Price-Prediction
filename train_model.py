import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


df = pd.read_csv("train.csv")


if 'id' in df.columns:
     df = df.drop('id',axis=1)

X = df.drop("price_range", axis=1)
y = df["price_range"]

feature_names = X.columns.tolist()
pickle.dump(feature_names, open("features.pkl", "wb"))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss'
)

model.fit(X_train_scaled, y_train)

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print(" Model trained and saved successfully!")