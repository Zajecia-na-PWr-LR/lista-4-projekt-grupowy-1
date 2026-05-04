import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, make_scorer

def load_and_split_neo_data(filepath, n_splits=1, test_size=0.2, random_state=42):
    df = pd.read_csv(filepath)
    
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    splits = gss.split(df, groups=df['id'])
    
    if n_splits == 1:
        train_idx, test_idx = next(splits)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, test_df
    else:
        def get_cv_folds():
            for train_idx, test_idx in splits:
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)
                yield train_df, test_df

        return get_cv_folds()

def custom_score(y_true, y_pred):
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
    tn, fp, fn, tp = cm.ravel()
    cost = (fp * 1) + (fn * 10**6)
    return cost

cost_score = make_scorer(custom_score, greater_is_better=False)
