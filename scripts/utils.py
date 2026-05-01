import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

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
