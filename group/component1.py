import pandas as pd
import json
import time
import os
import ollama

# Configuration
MODELS = ["mistral", "deepseek-r1", "phi4:14b"]
CATEGORIES = ["Revenue", "Expense", "Asset", "Liability", "Other"]
INPUT_FILE = "financial_transactions.csv"
JSON_OUTPUT_FILE = "categorized_transactions_output.json"
RAW_RESPONSES_FILE = "raw_model_responses.txt"

def build_prompt(description, amount, tx_type):
    """
    Constructs a standardized prompt for the LLMs.
    """
    return (
        f"Classify the following financial transaction as one of the following: {', '.join(CATEGORIES)}.\n"
        f"Description: '{description}'\n"
        f"Amount: ${amount}\n"
        f"Type: {tx_type}\n"
        "Only respond with a single category from the list above."
    )

def extract_category(response: str, model_name: str) -> str:
    """
    Extracts the predicted category from an LLM's full response.
    Handles model-specific delimiters (Deepseek-R1 uses <think>...</think>).
    """
    text = response
    # Deepseek-R1 special delimiter logic
    if model_name.lower() == "deepseek-r1" and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text_lower = text.lower()
    for category in CATEGORIES:
        if category.lower() in text_lower:
            return category
    return "Unknown"

def query_model(model_name: str, prompt: str):
    start = time.time()
    response = ollama.chat(
      model=model_name,
      messages=[{"role": "user", "content": prompt}]
    )
    runtime = time.time() - start
    raw = response["message"]["content"]
    predicted = extract_category(raw, model_name)
    return predicted, runtime, raw


def main():
    # Prepare raw responses log
    with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
        f.write("")

    # Load and clean data
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["transaction_id", "description", "amount", "type"])
    results = []

    # Process each transaction
    for _, row in df.iterrows():
        tid = row["transaction_id"]
        desc = row["description"].strip()
        amount = row["amount"]
        tx_type = row["type"].strip()

        prompt = build_prompt(desc, amount, tx_type)
        transaction_entry = {
            "transaction_id": tid,
            "description": desc,
            "amount": amount,
            "type": tx_type,
            "predictions": []
        }

        # Query each model
        for model in MODELS:
            predicted, runtime_sec, raw = query_model(model, prompt)

            # Log raw response for debugging
            with open(RAW_RESPONSES_FILE, "a", encoding="utf-8") as f:
                f.write(f"=== Transaction ID: {tid} ===\n")
                f.write(f"Model: {model}\n")
                f.write("Response:\n")
                f.write(raw + "\n")
                f.write("-" * 50 + "\n")

            # Append structured prediction
            transaction_entry["predictions"].append({
                "model": model.title(),
                "predicted_category": predicted,
                "runtime_sec": runtime_sec
            })

        results.append(transaction_entry)

    # Save structured JSON output
    with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {JSON_OUTPUT_FILE}")
    print(f"Full responses logged in {RAW_RESPONSES_FILE}")

if __name__ == "__main__":
    main()

# After execution, you'll have:
# 1. categorized_transactions_output.json — ready for Flask dashboard ingestion
# 2. raw_model_responses.txt — for detailed debugging of LLM behavior
