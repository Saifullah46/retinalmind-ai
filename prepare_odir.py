import os
import pandas as pd
import shutil

# =========================
# PATHS (UPDATE IF NEEDED)
# =========================
csv_file = r"D:\RetinalMind AI-Based Cognitive Risk Detection System\full_df.csv"
image_folder = r"D:\RetinalMind AI-Based Cognitive Risk Detection System\preprocessed_images"
output_folder = r"D:\RetinalMind AI-Based Cognitive Risk Detection System\data\train"

# =========================
# CREATE OUTPUT FOLDERS
# =========================
os.makedirs(os.path.join(output_folder, "0"), exist_ok=True)  # No Risk
os.makedirs(os.path.join(output_folder, "1"), exist_ok=True)  # Risk

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(csv_file)

print("Columns in CSV:", df.columns)  # Debug check

# =========================
# PROCESS FUNCTION
# =========================
def process_image(img_name, diagnosis):
    if pd.isna(img_name):
        return

    diagnosis = str(diagnosis).lower()

    # Mapping logic
    if "normal" in diagnosis:
        label = "0"   # No Risk
    else:
        label = "1"   # Risk

    src = os.path.join(image_folder, img_name)
    dst = os.path.join(output_folder, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

# =========================
# LOOP THROUGH DATA
# =========================
for _, row in df.iterrows():

    # LEFT IMAGE
    process_image(
        row.get('Left-Fundus'),
        row.get('Left-Diagnostic Keywords')
    )

    # RIGHT IMAGE
    process_image(
        row.get('Right-Fundus'),
        row.get('Right-Diagnostic Keywords')
    )

print("✅ Dataset Ready Successfully!")