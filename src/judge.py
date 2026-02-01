import pandas as pd
import re
import os

def rigorous_judge(video_name):
    csv_path = f"{video_name}_interpolated.csv"
    if not os.path.exists(csv_path):
        return 0.0, "Missing CSV"

    df = pd.read_csv(csv_path)
    
    # Validates Indian Format
    pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$')
    valid_plates = df['license_number'].apply(lambda x: bool(pattern.match(str(x))))
    
    # REWARD HACKING CHECK: Unique ID Variety
    unique_ratio = df['license_number'].nunique() / len(df) if len(df) > 0 else 0
    if unique_ratio < 0.05 and len(df) > 20:
        return 0.0, "Reward Hacking: Collapsed ID variety detected."

    return valid_plates.mean() * 100, "Success"