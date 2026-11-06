import os
import zipfile
import urllib.request

zip_filename = 'sat_math_csv_files.zip'
url = 'https://github.com/AllenDowney/SelectionSAT/raw/main/sat_math_csv_files.zip'

# Check if file exists, download if not
if not os.path.exists(zip_filename):
    print(f"Downloading {zip_filename}...")
    urllib.request.urlretrieve(url, zip_filename)
    print(f"Downloaded {zip_filename}")
else:
    print(f"{zip_filename} already exists, skipping download")

# Unzip the file
print(f"Extracting {zip_filename}...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('.')
    print(f"Extracted files: {', '.join(zip_ref.namelist())}")

