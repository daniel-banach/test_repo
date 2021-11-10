import os
import sys
from config import *

sys.path.append(f"{os.environ['MNT_DIR']}")

import joblib
import pandas as pd



def load_model():
    model = joblib.load(open(os.environ['MNT_DIR'] + '/' + MODEL, 'rb'))
    return model

def load_features():
    features = joblib.load(open(os.environ['MNT_DIR']+'/' +FEATURES,'rb' ))
    return features

def predict(event, context):
    model = load_model()
    features = load_features()
    features = list(set(features)-set('price'))
    data = pd.DataFrame(event['data'])[features]
    return model.predict(data)
