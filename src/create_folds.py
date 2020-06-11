import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df = df.dropna().reset_index(drop=True)
    df["kfold"] = -1

    df = df.sample(frac=1, random_state=100).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=8)

    for fold, (train, valid) in enumerate(kf.split(X=df, y=df.sentiment.values)):
        print(len(train), len(valid))
        df.loc[valid, "kfold"] = fold

    df.to_csv("../input/train_folds_8.csv", index=False)
