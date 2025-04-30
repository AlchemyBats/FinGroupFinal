import zipfile
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

zip_path = "financial_transactions.csv 1.zip"
extract_path = "financial_transactions"

if not os.path.exists(extract_path):
    os.makedirs(extract_path)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

csv_file_path = os.path.join(extract_path, 'financial_transactions.csv')

print("Loading dataset...")
df = pd.read_csv(csv_file_path)

def assign_category(row):
    if row['type'] == 'credit':
        return 'Revenue'
    elif row['type'] == 'debit' and row['amount'] < 5000:
        return 'Expense'
    else:
        return 'Asset'

print("Processing ground truth categories...")
df['ground_truth_category'] = df.apply(assign_category, axis=1)

categories = ['Revenue', 'Expense', 'Asset']

def mock_model_predict(model_name, data):
    np.random.seed(hash(model_name) % 2**32)
    predictions = []
    
    start_time = time.time()
    for _, row in data.iterrows():
        true_category = row['ground_truth_category']
        
        if model_name == "Deepseek-R1":
            if np.random.rand() < 0.9:
                predictions.append(true_category)
            else:
                predictions.append(np.random.choice([c for c in categories if c != true_category]))

        elif model_name == "Phi-4":
            if np.random.rand() < 0.85:
                predictions.append(true_category)
            else:
                predictions.append(np.random.choice([c for c in categories if c != true_category]))

        elif model_name == "Mistral":
            if np.random.rand() < 0.8:
                predictions.append(true_category)
            else:
                predictions.append(np.random.choice([c for c in categories if c != true_category]))

        else:
            predictions.append(np.random.choice(categories))

    end_time = time.time()
    elapsed_time = end_time - start_time

    return predictions, elapsed_time

def evaluate_models(models, data):
    results = []

    for model in models:
        preds, elapsed_time = mock_model_predict(model, data)
        correct = sum([p == t for p, t in zip(preds, data['ground_truth_category'])])
        accuracy = correct / len(data)
        results.append({
            'Model': model,
            'Accuracy': accuracy,
            'Time_Seconds': elapsed_time
        })
    
    return pd.DataFrame(results)

models = ['Deepseek-R1', 'Phi-4', 'Mistral']

print("Evaluating models...")
metrics_df = evaluate_models(models, df)
print(metrics_df)

plt.figure(figsize=(8, 5))
plt.bar(metrics_df['Model'], metrics_df['Accuracy'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(metrics_df['Model'], metrics_df['Time_Seconds'])
plt.title('Model Speed Comparison')
plt.xlabel('Model')
plt.ylabel('Time Taken (Seconds)')
plt.grid(axis='y')
plt.show()
