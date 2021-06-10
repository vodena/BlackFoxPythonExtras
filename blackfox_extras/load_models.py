from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow
import joblib
import xgboost
import os

def load_model(path):
    try:
        model = tensorflow.keras.models.load_model(path)
        return model
    except:
        try:
            model = joblib.load(path)
            return model
        except:
            try:
                model = xgboost.XGBClassifier()
                model.load_model(path)
                return model
            except:
                try:
                    model = xgboost.XGBRegressor()
                    model.load_model(path)
                    return model
                except:
                    try:
                        model = MultiOutputRegressor()
                        model.load_model(path)
                        return model
                    except:
                        raise Exception('An error occurred during model loading.')