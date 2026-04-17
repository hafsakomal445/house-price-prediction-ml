import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(data_path):
    df = pd.read_csv(data_path)

    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Save model
    joblib.dump(model, 'models/model.pkl')

    # ✅ ADD THIS LINE (IMPORTANT)
    joblib.dump(X.columns.tolist(), 'models/features.pkl')

    return model

if __name__ == "__main__":
    train_model('data/processed_data.csv')