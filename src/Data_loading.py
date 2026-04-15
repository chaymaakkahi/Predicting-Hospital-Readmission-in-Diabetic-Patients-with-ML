# ================================================================
# data/download_data.py
# Downloads the Diabetes 130-US Hospitals dataset from UCI
# ================================================================

import urllib.request
import zipfile
import os

URL      = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
ZIP_PATH = "data/diabetes.zip"
OUT_DIR  = "data/"

print("Downloading dataset from UCI...")
urllib.request.urlretrieve(URL, ZIP_PATH)

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(OUT_DIR)

os.remove(ZIP_PATH)
print(f"✓ Dataset saved to {OUT_DIR}")
print("  Files:", os.listdir(OUT_DIR))
