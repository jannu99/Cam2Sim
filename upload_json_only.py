from datasets import load_dataset
from huggingface_hub import login

# 1. Login with your token
# (This is the token you shared in previous scripts)
login()

# 2. Load the dataset
print("⏳ Downloading dataset info...")
ds = load_dataset("jannu99/gurickestraemp4_25_12_01", split="train")

# 3. Verify Columns
print("\n✅ COLUMNS FOUND:", ds.column_names)

# 4. Check the content of the first row
first_item = ds[0]
print("\n🔎 SAMPLE ENTRY:")
print(f"   - Image: {first_item['image']}")
print(f"   - Previous: {first_item['previous']}")  # <--- This should be here!
print(f"   - Text: {first_item['text']}")