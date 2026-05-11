from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, make_scorer

from scripts.preprocess_neo import preprocess_neo

def load_and_split_neo_data(filepath, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    splits = gss.split(df, groups=df['id'])
    
    train_idx, test_idx = next(splits)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df

def load_preprocess_split(filepath: Path):
    df_raw = pd.read_csv(filepath)
    df_clean = preprocess_neo(df_raw)
    neo_path = filepath.parents[0] / "neo_preprocessed.csv"
    df_clean.to_csv(neo_path, index=False)

    train_df, test_df = load_and_split_neo_data(neo_path)
    return train_df, test_df

def custom_score(y_true, y_pred):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    tn, fp, fn, tp = cm.ravel()
    cost = (fp * 1) + (fn * 10**6)
    return cost

cost_score = make_scorer(custom_score, greater_is_better=False)
