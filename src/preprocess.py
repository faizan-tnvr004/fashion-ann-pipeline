import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Load parameters
with open("params.yaml") as f:
    params = yaml.safe_load(f)["preprocess"]

os.makedirs('data/processed', exist_ok=True)

# Load raw data
x_train_raw = np.load('data/raw/x_train.npy')
y_train = np.load('data/raw/y_train.npy')
x_test_raw = np.load('data/raw/x_test.npy')
y_test = np.load('data/raw/y_test.npy')

# Normalize to [0, 1]
x_train_norm = (x_train_raw - 127.5) / 127.5
x_test = x_test_raw / 255.0

# Split train/validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train_norm, y_train, 
    test_size=params['test_size'], 
    random_state=params['seed']
)

# Save processed data
np.save('data/processed/x_train.npy', x_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/x_val.npy', x_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/x_test.npy', x_test)
np.save('data/processed/y_test.npy', y_test)

print("Processed data saved to data/processed/")