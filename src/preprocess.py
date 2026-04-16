import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop columns
    df = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence'], errors='ignore')

    # Fill missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

    # Encoding
    df = pd.get_dummies(df, drop_first=True)

    return df