#%% VSCode Notebook
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
#%%
df_train = pd.read_excel('./data/bag_of_words_vec_train.xlsx')
df_validation = pd.read_excel('./data/bag_of_words_vec_validation.xlsx')
# %%
