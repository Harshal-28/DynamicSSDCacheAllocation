import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_on_trace(trace_file):
    print(f"Loading {trace_file}...")
    df = pd.read_csv(trace_file)

    # 1. Create the Ground Truth Labels (Lookahead window of 200 requests)
    window_size = 200
    reused = np.zeros(len(df))
    last_seen = {}

    # Iterate backwards to find the next access distance
    for i in range(len(df)-1, -1, -1):
        file = df['file'].iloc[i]
        if file in last_seen and (last_seen[file] - i) <= window_size:
            reused[i] = 1
        last_seen[file] = i

    df['will_be_reused'] = reused
    df['is_peak_hour'] = df['hour'].apply(lambda h: 1 if 9 <= h <= 17 else 0)

    # 2. Compute historical category hit rates
    cat_counts = df['category'].value_counts()
    cat_reused = df[df['will_be_reused'] == 1]['category'].value_counts()
    cat_rates = (cat_reused / cat_counts).fillna(0.5).to_dict()
    df['category_hit_rate'] = df['category'].map(cat_rates)

    # 3. Train Logistic Regression
    X = df[['is_peak_hour', 'category_hit_rate', 'user_priority']]
    y = df['will_be_reused']

    model = LogisticRegression()
    model.fit(X, y)

    # 4. Output Weights
    print("========================================")
    print("EXTRACTED MODEL WEIGHTS FOR SIMULATOR")
    print("========================================")
    print(f"'intercept': {model.intercept_[0]:.4f}")
    print(f"'time_factor': {model.coef_[0][0]:.4f}")
    print(f"'category_rate': {model.coef_[0][1]:.4f}")
    print(f"'user_priority': {model.coef_[0][2]:.4f}")
    print("========================================")

if __name__ == "__main__":
    train_on_trace('access_trace.csv')