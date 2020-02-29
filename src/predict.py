import os
import pandas as  pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")



def predict():
    df = pd.read_csv(TEST_DATA)   
    predictions = None
    
    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models",  f"models/{MODEL}_{FOLD}_label_encoder.pkl"))
        for c in train_df.columns:
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())

        # data is ready to train
        clf = joblib.load(os.path.join("models",  f"models/{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models",  f"models/{MODEL}_{FOLD}_columns.pkl"))
        # clf.fit(train_df,ytrain)
        preds = clf.predict_proba(valid_df)[:,1]
        # print(metrics.roc_auc_score(yvalid,preds))
        # print(preds)
     
if __name__ == "__main__":