import os
import pandas as pd
from sklearn import model_selection

if __name__ =="__main__":
    input_path = "."
    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    df["Kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    Kf = model_selection.StratifiedKFold(n_splits=10)
    for fold_, (_,_) in enumerate(Kf.split(X=df, y=y)):
        df.loc[:, "Kfold"] = fold_
    df.to_csv(os.path.join(input_path, "train.csv"), index=False)