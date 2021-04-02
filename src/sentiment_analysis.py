#%% VSCode Notebook
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
tqdm.pandas()
#%%
df_train = pd.read_excel('./data/bag_of_words_vec_train.xlsx')
X_train = np.array(list(df_train['lemma_vector_json'].progress_apply(lambda x: np.array(json.loads(x)))))
y_train = np.array(df_train['Userscore'])
del df_train # Free up memory
# %%
df_validation = pd.read_excel('./data/bag_of_words_vec_validation.xlsx')
X_validation = np.array(list(df_validation['lemma_vector_json'].progress_apply(lambda x: np.array(json.loads(x)))))
y_validation = np.array(df_validation['Userscore'])
del df_validation # Free up memory

#%%
model = LinearRegression()
model.fit(X_train, y_train)

# %%
y_validation_predictions = np.array(model.predict(X_validation))
# %%
# Normalize
y_validation_predictions_norm = y_validation_predictions
y_validation_predictions_norm[y_validation_predictions_norm<1] = 1
y_validation_predictions_norm[y_validation_predictions_norm>10] = 10
#%%

validation_rmse = mean_squared_error(y_validation,y_validation_predictions_norm,squared=False)
print(f"Validation RMSE: {validation_rmse}")
# %%
