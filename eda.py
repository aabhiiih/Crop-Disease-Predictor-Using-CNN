import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def generate_eda(crops, user_id):
    if not crops:
        return None
    
    # Extract status/disease robustly
    predictions = []
    for c in crops:
        parts = c.prediction.split(',')
        if len(parts) > 1:
            # Healthy or Sick case: "Leaf: X, Status: Y (Z%)" or "Leaf: X, Sick, Y (Z%)"
            status_part = parts[1].split('(')[0].strip()
            predictions.append(status_part if status_part != 'Status: Healthy' else 'Healthy')
        else:
            # Unknown case: "Unknown: X (Y%)"
            predictions.append('Unknown')

    df = pd.DataFrame({'Prediction': predictions})
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Prediction')
    plt.title(f'EDA for User {user_id}: Prediction Distribution')
    plt.xticks(rotation=45, ha='right')
    eda_path = f'static/eda/eda_{user_id}.png'
    os.makedirs(os.path.dirname(eda_path), exist_ok=True)
    plt.savefig(eda_path)
    plt.close()
    return eda_path