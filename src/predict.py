import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np


import config
import dispatcher

# TRAININF_DATA=os.environ.get('TRAINING_DATA')
# TEST_DATA=os.environ.get('TEST_DATA')
# FOLD=os.environ.get("FOLD")
# MODEL = os.environ.get("MODEL")

TRAINING_DATA=config.TRAINING_DATA
TEST_DATA=config.TEST_DATA
FOLD = config.FOLD
MODEL = config.MODEL

FOLD_MAPPING = {
    0:[1,2,3,4],
    1:[0,2,3,4],
    2:[0,1,3,4],
    3:[0,1,2,4],
    4:[0,1,2,3]
}

# if __name__=="__main__":
def predict():
    df=pd.read_csv(TEST_DATA)
    test_idx= df["id"].values
    predictions=None

    # label_encoders = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
    # clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
    # cols = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_columns.pkl"))

        for c in cols:
            print(c)
            lbl =  encoders[c]
            df.loc[:,c]=lbl.transform(df[c].values.tolist())
        # Data ready to train
        clf = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(valid_df)[:, 1]
      
        if FOLD == 0:
            predictions = preds
        else:
            predictions+=preds

    predictions/=5

    sub = pd.DataFrame(np.columns_stack((test_idx,predictions)),columns=["id","target"])
    return sub
if __name__=="__main__":
    submission = predict()
    submission.to_csv(f"model/{MODEL}.csv",index=False)