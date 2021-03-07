#%% VSCODE Notebook
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

#%% If using colab
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_PATH = "/content/drive/Shared drives/CMPUT466 Project"

#%% If using windows
DRIVE_PATH = Path("G:/Shared drives/CMPUT466 Project")
#%%
DATA_FOLDER=DRIVE_PATH/"src/data/"
train_set = pd.read_csv(DATA_FOLDER/'user_data_train.csv')
test_set = pd.read_csv(DATA_FOLDER/'user_data_test.csv')
validation_set = pd.read_csv(DATA_FOLDER/'user_data_validation.csv')
full_set = pd.concat([train_set,test_set,validation_set])
# %%
mean_rating = train_set['Userscore'].mean()
full_title_groups = full_set.groupby('Title')
titles = [x[0] for x in full_title_groups]
mean_ratings = {x[0]:x[1]['Userscore'].mean() for x in full_title_groups}
# base_vector_row = {x:(mean_ratings[x] if x in mean_ratings else mean_ratings) for x in full_set}
#%%
def vectorize_users(df:pd.DataFrame):
    user_groups = df.groupby('Username')
    vector_rows = []
    for name, group in user_groups:
        vector_row = {}
        for index, row in group.iterrows():
            vector_row[row['Title']] = row['Userscore']
        vector_rows.append(vector_row)
    return pd.DataFrame(vector_rows)
        
vec_df = vectorize_users(train_set)
# %%
# get_closest_user based on https://codereview.stackexchange.com/a/134918
vec_columns_set = set(vec_df.columns)
top_n = 10

def get_closest_value(user_vec_df, title):
    title_intersection = list(set(user_vec_df.columns).intersection(vec_columns_set))
    user_vec = user_vec_df[title_intersection].to_numpy()[0]
    filtered_vec_df = vec_df[pd.notnull(vec_df[title])]
    for column in title_intersection:
        filtered_vec_df[column] = filtered_vec_df[column].fillna(mean_ratings[column])
    possible_match_vectors = filtered_vec_df[title_intersection].to_numpy()
    distances = distance.cdist([user_vec], possible_match_vectors, 'euclidean')[0]
    min_val = distances.min()
    closest_indexes = [i for i, x in enumerate(distances) if np.isclose(x,min_val)]
    top_n_max = min(top_n,len(distances))
    if len(closest_indexes)<top_n_max:
        closest_indexes = np.argpartition(distances, top_n_max-1)[-top_n_max:]
    return filtered_vec_df.iloc[closest_indexes][title].mean()
#%%

# Note assumes that games across different platforms are the same
def get_prediction_df(df):
    user_groups = df.groupby('Username')
    rows_with_predictions = []
    for name, group_df in tqdm(user_groups):
        for index, row in group_df.iterrows():
            title = row['Title']
            prediction = 0
            if title in vec_df.columns:
                user_vec_df = vectorize_users(group_df.drop(index))
                prediction = get_closest_value(user_vec_df, title)
            else:
                # If not seen before use global mean
                prediction = mean_rating
            row['PredictedScore'] = prediction
            rows_with_predictions.append(row)
    return pd.DataFrame(rows_with_predictions)

# %%
validation_predictions = get_prediction_df(validation_set)
validation_rmse = mean_squared_error(validation_predictions['Userscore'], validation_predictions['PredictedScore'], squared = False)
print(f"Validation RMSE {validation_rmse}")
# %%
tests_predictions = get_prediction_df(test_set)
test_rmse = mean_squared_error(tests_predictions['Userscore'], tests_predictions['PredictedScore'], squared = False)
print(f"Test RMSE {test_rmse}")
# %%
validation_predictions['ScoreDiff'] = validation_predictions['PredictedScore'] - validation_predictions['Userscore']
# %%
validation_predictions['BaseDiff'] = 7.78094579978368
# %%
mean_squared_error(validation_predictions['Userscore'], validation_predictions['BaseDiff'], squared = False)
# %%
validation_predictions[['Username','Userscore','PredictedScore','ScoreDiff']].head
# %%
