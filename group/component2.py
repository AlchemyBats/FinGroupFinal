import ollama
import pandas as pd
import json

# Define models and files
models = ["phi4:14b", "mistral", "deepseek-r1"]  # Correct Ollama model identifiers
csv_file = "corporate_market_data.csv"
output_file = "anomaly_results.json"

# Load data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error loading {csv_file}: {e}")
    exit(1)
data_text = df.to_string(index=False)

# Build prompt
prompt = (
    f"\n\nDataset:\n{data_text}"
    "Analyze the previous dataset and identify any financial anomalies. "
    "For each anomaly, provide transaction_id, reason, and confidence_score if available."
)

# Run models and collect results
results = {"model_results": []}
for model in models:
    # Attempt to pull the model
    try:
        ollama.pull(model)
    except Exception as e:
        print(f"Warning: Could not pull model {model}: {e}")

    # Execute anomaly detection
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]
    except Exception as e:
        print(f"Error with model {model}: {e}")
        results["model_results"].append({
            "model": model,
            "error": str(e)
        })
        continue

    # Attempt to parse JSON; fallback to raw text
    try:
        parsed = json.loads(content)
        parsed.setdefault("model", model)
    except json.JSONDecodeError:
        parsed = {"model": model, "raw_output": content}
    results["model_results"].append(parsed)

# Save to JSON
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Anomaly results saved to {output_file}")
