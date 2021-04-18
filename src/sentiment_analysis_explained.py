#%% Vscode notebook - Explaining sentiment analysis model
import json
import numpy as np
import shap
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
#%%
df_test = pd.read_excel('./data/bag_of_words_vec_test.xlsx')
X_test = np.array(list(df_test['lemma_vector_json'].progress_apply(lambda x: np.array(json.loads(x)))))
y_test = np.array(df_test['Userscore'])
del df_test # Free up memory

#%%
with open('./data/bag_of_words_tokens.json') as token_file:
    token_map = np.array(json.loads(token_file.read()))

#%%

def load_model(path):
    with open(path,'r') as model_file:
        model_dict = json.loads(model_file.read())
    model = LinearRegression()
    model.coef_ = np.array(model_dict['coef'])
    model.intercept_ = np.array(model_dict['intercept'])
    return model

model = load_model('./models/sentiment_bow_1.json')
# %%
y_test_predictions = np.array(model.predict(X_test))
# %%
# Normalize
y_test_predictions_norm = y_test_predictions
y_test_predictions_norm[y_test_predictions_norm<0] = 0
y_test_predictions_norm[y_test_predictions_norm>10] = 10
#%%
test_rmse = mean_squared_error(y_test,y_test_predictions_norm,squared=False)
print(f"Test RMSE: {test_rmse}")
#%% Explain
explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)
#%%
shap_values_normalized = shap_values
shap_values_normalized[shap_values_normalized>10]=10
shap_values_normalized[shap_values_normalized<-10]=-10
shap_values_normalized_abs = np.absolute(shap_values_normalized)
feature_mean = np.mean(shap_values_normalized_abs,axis=0)
top_25_feature_indexes = np.argsort(feature_mean)[::-1][:25]
#%%
X_test_top_features = np.take(X_test,top_25_feature_indexes,axis=1)
shap_values_top_features = np.take(shap_values_normalized,top_25_feature_indexes,axis=1)
shap.summary_plot(shap_values_top_features, X_test_top_features, feature_names=token_map[top_25_feature_indexes])
# %%
y_diff = np.abs(y_test-y_test_predictions_norm)
top_diff_indexes = np.argsort(y_diff)[::-1][:50]
# %%
# shap.initjs()
index = top_diff_indexes[0]
select = X_test[index]>0
shap.force_plot(
    explainer.expected_value, shap_values[index,][select], X_test[index][select],
    feature_names=token_map[select],matplotlib=True
)
# %%
