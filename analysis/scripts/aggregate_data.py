### Aggregates the data across sessions that have opted in for full permissions ###
# %%
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from paths import DATA_DIR, DATA_ROOT
from text_embeddings import add_embeddings
# %%
p = pd.read_csv(DATA_ROOT / "permissions.csv")
#%%
# get the sessions in which both participants have selected level 1
p = p[p["Level 1"] == 'X']
kpids = []
for i in p['Pair ID'].unique():
    if len(p[p['Pair ID'] == i]) == 2:
        kpids.append(i)
p = p[p['Pair ID'].isin(kpids)]
# # %%
df = []
ddir = DATA_DIR
dpaths = [Path(i / "data_v3.csv") for i in list(ddir.glob("0*")) if i.is_dir() and int(i.name) in kpids]
# %%
# concat all the csvs together
for dpath in dpaths:
    try:
        df.append(pd.read_csv(dpath))
    except Exception as e:
        print(f"Error reading {dpath}: {e}")
df = pd.concat(df)
# %%
# filter down to only rows that contain at least four words
df = df[df['text'].str.split().str.len() >= 4]
# %%
# add the embeddings
df = add_embeddings(df)
df.to_csv(DATA_ROOT / "aggregated_data_with_embeddings.csv", index=False)