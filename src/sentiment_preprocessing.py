#%% VS Code Notebook

import json
import operator
import os
import numpy as np
import spacy
import pandas as pd

from collections import Counter
from functools import reduce,partial
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
tqdm.pandas()
#%%
def load_df(dir_path):
    dfs = []
    dir_full_path = Path(dir_path)
    for file in os.listdir(dir_full_path):
        dfs.append(pd.read_csv(dir_full_path/file))
    return pd.concat(dfs)

df_train = load_df('./data/train')
df_train['Comment'].fillna('',inplace=True)
df_test = load_df('./data/test')
df_test['Comment'].fillna('',inplace=True)
df_validation = load_df('./data/validation')
df_validation['Comment'].fillna('',inplace=True)
# %%
# python -m spacy download en_core_web_sm
# spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")

def get_counter_from_comment(comment):
    doc = nlp(comment)
    lemmas = [x.lemma_ for x in doc]
    # lemmas = [x.lemma for x in doc]
    # lemmas_without_stop_tokens = [x.lemma_ for x in doc if not x.is_stop]
    # simple_tags = [x.pos_ for x in doc]
    # specific_tags = [x.tag_ for x in doc]
    return Counter(lemmas)
    # return Counter(lemmas_without_stop_tokens)
    # return Counter(simple_tags)
    # return Counter(specific_tags)

def parallel_progress_apply(series):
    return process_map(get_counter_from_comment, list(series), max_workers=8)

# Need to have these annoying main process checks for parallel to work (Not offical supported in juyter notebooks) 
if __name__ == '__main__':
    df_train['lemma_counter'] = parallel_progress_apply(df_train['Comment'])
# %%
if __name__ == '__main__':
    df_validation['lemma_counter'] = parallel_progress_apply(df_validation['Comment'])
# %%
reduce_add = partial(reduce,operator.add)

def parallel_progress_reduce(series):
    groups = np.array_split(series, 512)
    reduced_groups = process_map(reduce_add, groups, max_workers=8)
    return reduce_add(tqdm(reduced_groups))

if __name__ == '__main__':
    train_counter=parallel_progress_reduce(df_train['lemma_counter'])
# %%
if __name__ == '__main__':
    occurence_threshold = 50
    train_counter_filtered = {x:count for x,count in train_counter.items() if count>=occurence_threshold}
# %%
if __name__ == '__main__':
    vec_tokens = list(train_counter_filtered.keys())

    # TODO: We might want to add normalization here
    def counter_to_vec(counter):
        return json.dumps([counter.get(x,0) for x in vec_tokens])

    df_train['lemma_vector_json'] = df_train['lemma_counter'].progress_apply(counter_to_vec)
    df_validation['lemma_vector_json'] = df_validation['lemma_counter'].progress_apply(counter_to_vec)
# %%
if __name__ == '__main__':
    selected_columns = ['Title', 'Platform', 'Username', 'Userscore', 'lemma_vector_json']
    df_train[selected_columns].to_excel('./data/bag_of_words_vec_train.xlsx')
    df_validation[selected_columns].to_excel('./data/bag_of_words_vec_validation.xlsx')
# %%
