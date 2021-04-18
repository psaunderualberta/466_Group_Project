#%% VSCode Notebook
import json
import numpy as np
import pandas as pd
import tensorflow as tf

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
# %%
df_test = pd.read_excel('./data/bag_of_words_vec_test.xlsx')
X_test = np.array(list(df_test['lemma_vector_json'].progress_apply(lambda x: np.array(json.loads(x)))))
y_test = np.array(df_test['Userscore'])
del df_test # Free up memory

#%% Reducing features did not seem to imporve performance
# train_counts = np.sum(X_train,axis=0)
# train_count_indexes = [i for i,x in enumerate(train_counts) if x>=100]
# #%%
# X_train = X_train[:,train_count_indexes]
# X_validation = X_validation[:,train_count_indexes]
# X_test = X_test[:,train_count_indexes]
#%%
tf.random.set_seed(
    314
)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(10470,)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
epochs = 1
# history = model.fit(
model.fit(
    np.concatenate([X_train,X_validation]),
    np.concatenate([y_train,y_validation]),
    # validation_data=(X_validation, y_validation),
    epochs=epochs
)
# # %% Check if it is overfitting to train set
# # Copied from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
# import matplotlib.pyplot as plt

# acc = history.history['mean_squared_error']
# val_acc = history.history['val_mean_squared_error']

# loss = history.history['mean_squared_error']
# val_loss = history.history['val_mean_squared_error']

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
# %%
y_validation_predictions = np.array(model.predict(X_validation))
y_validation_predictions_norm = y_validation_predictions
y_validation_predictions_norm[y_validation_predictions_norm<0] = 0
y_validation_predictions_norm[y_validation_predictions_norm>10] = 10
#%%

validation_rmse = mean_squared_error(y_validation,y_validation_predictions_norm,squared=False)
print(f"Validation RMSE: {validation_rmse}")
y_validation_predictions = np.array(model.predict(X_validation))
# %%
y_test_predictions = np.array(model.predict(X_test))
y_test_predictions_norm = y_test_predictions
y_test_predictions_norm[y_test_predictions_norm<0] = 0
y_test_predictions_norm[y_test_predictions_norm>10] = 10
#%%

test_rmse = mean_squared_error(y_test,y_test_predictions_norm,squared=False)
print(f"Test RMSE: {test_rmse}")
# %%
