#%% VSCODE Notebook
import os
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

#%% If using colab
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_PATH = "/content/drive/Shared drives/CMPUT466 Project"

#%% If using windows
DRIVE_PATH = Path("G:/Shared drives/CMPUT466 Project")

#%%
GAME_DATA_PATH = DRIVE_PATH/"src/data/raw_metacritic_game_info.csv"
USER_DATA_PATH = DRIVE_PATH/"src/data/raw_metacritic_game_user_comments.csv"
#%%
user_data = pd.read_csv(USER_DATA_PATH)
# Remove anonymous users
user_dat = user_data[user_data["Username"]!="[Anonymous]"]
#game_data = pd.read_csv(GAME_DATA_PATH)
#%%
user_data
#%%
user_groups = user_data.groupby('Username')
# %%
# 70 - 10 - 20 Split
train_validation_groups, test_groups = train_test_split(
    list(user_groups), 
    train_size=.8,
    random_state=314
)
train_groups, validation_groups = train_test_split(
    train_validation_groups, 
    train_size=.875,
    random_state=314159
)
# for name, group in user_groups:
#     group
# %%
train_set = pd.concat([x[1] for x in train_groups])
test_set = pd.concat([x[1] for x in test_groups])
validation_set = pd.concat([x[1] for x in validation_groups])
# %%
train_count = len(train_set)
test_count = len(test_set)
validation_count = len(validation_set)
total_count = train_count+test_count+validation_count
print(f"Train: {train_count} ({train_count/total_count})")
print(f"Test: {test_count} ({test_count/total_count})")
print(f"Validation: {validation_count} ({validation_count/total_count})")
# %%
DATA_FOLDER=DRIVE_PATH/"src/data/"
train_set.to_csv(DATA_FOLDER/'user_data_train.csv',index=False)
test_set.to_csv(DATA_FOLDER/'user_data_test.csv',index=False)
validation_set.to_csv(DATA_FOLDER/'user_data_validation.csv',index=False)
# %%
