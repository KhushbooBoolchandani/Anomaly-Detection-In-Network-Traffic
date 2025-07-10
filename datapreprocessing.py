import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate","label"
]

# Load and save as clean CSV
df = pd.read_csv("C:/Users/khush/OneDrive/Documents/kddcup.data_10_percent_corrected.csv",names=columns)
print("Dataset loaded:", df.shape)

# Drop duplicate records
df.drop_duplicates(inplace=True)
print("After removing duplicates:", df.shape)

# Drop missing values if any (none expected in this dataset)
df.dropna(inplace=True)
print("After removing missing values:", df.shape)

# Encode categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder  # store encoder for future use if needed

# Convert label to binary: 0 = normal, 1 = attack
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
print("Label distribution:\n", df['label'].value_counts())

# Drop constant column (optional but common for this dataset)
if df['num_outbound_cmds'].nunique() == 1:
    df.drop(columns=['num_outbound_cmds'], inplace=True)

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Combine scaled features and target back into a single DataFrame
df_final = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# Save the final preprocessed dataset
df_final.to_csv("kddcup_clean.csv", index=False)
print("Preprocessed dataset saved as 'kddcup_clean.csv'")
print("Final shape:", df_final.shape)

import os
from joblib import dump

os.makedirs("models", exist_ok=True)
dump(scaler, "models/scaler.joblib")
print("✅ Scaler saved to models/scaler.joblib")