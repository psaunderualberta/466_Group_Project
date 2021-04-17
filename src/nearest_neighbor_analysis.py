#%%
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

#%%
train_df = pd.read_csv('data/user_data_train_no_comments.csv') 
valid_df = pd.read_csv('validation_with_predictions.csv') 
test_df = pd.read_csv('test_with_predictions.csv')

#%%
train_df.groupby('Username').ngroups

# %%
train_full_df = pd.concat([train_df,valid_df])
def get_score(x):
    game_score = train_full_df[train_full_df['Game_ID']==x['Game_ID']]['Userscore'].mean()
    if pd.isnull(game_score):
        return train_full_df['Userscore'].mean()
    return game_score
valid_df['BaselineScore'] = valid_df.apply(get_score,axis=1)
# test_df['BaselineScore'] = test_df.apply(get_score,axis=1)

# train_df[train_df['Game_ID']==0]['Userscore'].mean()
# %%
baselines_rmse = mean_squared_error(test_df['Userscore'], test_df['BaselineScore'], squared = False)
print(baselines_rmse)
# %%
valid_df[pd.isnull(valid_df['BaselineScore'])]
# %%
valid_df['DiffScore'] = valid_df['BaselineScore'] - valid_df['PredictedScore']
# %%
valid_df['DiffScore'].max()

# %%
ax = valid_df['PredictedScore'].hist(bins=np.arange(12)-0.5)
ax.set_xlabel('Predicted Userscore')
ax.set_ylabel('Number of Reviews')
# ax.set_title('Histogram of Predicted Userscores from Validation Dataset')
ax.set_xticks(np.arange(11))
ax.grid(False)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig('predicted_userscore_histogram_validation_data.png', dpi=600)
# %%
ax = valid_df['BaselineScore'].hist(bins=np.arange(12)-0.5)
ax.set_xlabel('Baseline Userscore')
ax.set_ylabel('Number of Reviews')
# ax.set_title('Histogram of Predicted Userscores from Validation Dataset')
ax.set_xticks(np.arange(11))
ax.grid(False)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig('baseline_userscore_histogram_validation_data.png', dpi=600)
# %%
ax = valid_df['Userscore'].hist(bins=np.arange(12)-0.5)
ax.set_xlabel('Actual Userscore')
ax.set_ylabel('Number of Reviews')
# ax.set_title('Histogram of Predicted Userscores from Validation Dataset')
ax.set_xticks(np.arange(11))
ax.grid(False)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig('actual_userscore_histogram_validation_data.png', dpi=600)

# %%
