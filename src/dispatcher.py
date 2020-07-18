from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics



MODELS = {
    "randomforest":ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,verbose=2),
    "extratress":ensemble.ExtraTreesClassifier(n_estimators=200,n_jobs=-1,verbose=2)
}