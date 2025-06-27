

import pandas as pd
from google.colab import files
file_path = "raw_data_set.csv"
df = pd.read_csv(file_path)


df.head()

import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["instruction_clean"] = df["instruction"].apply(clean_text)
df["response_clean"] = df["response"].apply(clean_text)


df_cleaned = df[["instruction_clean", "response_clean"]].dropna().drop_duplicates()
df_cleaned = df_cleaned[df_cleaned["instruction_clean"] != ""]
df_cleaned = df_cleaned[df_cleaned["response_clean"] != ""]


cleaned_csv_path = "/content/cleaned_customer_support.csv"
df_cleaned.to_csv(cleaned_csv_path, index=False)

df=pd.read_csv("/content/cleaned_customer_support.csv")
df.head()




# Convert to prompt-response format
def format_instruction_response(row):
    return f"### Customer Query:\n{row['instruction_clean']}\n\n### Response:\n{row['response_clean']}"

# Write to text file
with open("customer_support_clean.txt", "w", encoding="utf-8") as f:
    for line in df.apply(format_instruction_response, axis=1):
        f.write(line + "\n\n")



