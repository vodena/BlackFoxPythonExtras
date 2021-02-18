# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:58:20 2020

@author: VODENA06
"""

# Importing Black Fox service libraries

from blackfox import BlackFox
from blackfox import AnnOptimizationConfig, RandomForestOptimizationConfig, XGBoostOptimizationConfig
from blackfox import AnnOptimizationEngineConfig, OptimizationEngineConfig
from blackfox import LogWriter, CsvLogWriter
from blackfox_extras import prepare_input_data, scale_output_data
from blackfox import InputConfig, Encoding
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
mpl.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import h5py
import time

# blackfox_url = 'http://147.91.204.14:32702' # BF_URL
# blackfox_url = 'http://147.91.204.85:32723'# ~VODAFONE_URL
blackfox_url = 'http://147.91.204.14:32706' # BF FOR MAPE verziju blackfox 3.2.0
# blackfox_url = 'http://localhost:5000'
bf = BlackFox(blackfox_url)


dataframe = pd.read_csv('F:\\.spyder-py3\\projekti\\Vodena_Primeri\\ImplementingEncodingInBf\\data\\data_set.csv')


y = dataframe.iloc[:, -1:].values
X = dataframe.iloc[:, :-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)


start = time.time()

ec = AnnOptimizationEngineConfig(
    population_size = 50,
    max_num_of_generations = 10
)

c = AnnOptimizationConfig(
    problem_type = "BinaryClassification",
    binary_optimization_metric = "ROC_AUC",
    engine_config = ec,
    inputs=[
        InputConfig(encoding='OneHot', is_optional=True),
        InputConfig(encoding='Dummy', is_optional=True),
        InputConfig(encoding='Target', is_optional=True),
        InputConfig(encoding='Effect', is_optional=True)
    ]
    )

# Use CTRL + C to stop optimization
(ann_io, ann_info, ann_metadata) = bf.optimize_ann(
    input_set = X_train,
    output_set = y_train,
    config = c,
    model_path = 'test_model_encoding_version_bf_feature_selection_f.h5',
    delete_on_finish = False
)

end = time.time()
time_BF_ann = int(end-start)

print('\nann info:')
print(ann_info)

print('\nann metadata:')
print(ann_metadata)



# Get model metadata
model_name = 'F:\\.spyder-py3\\projekti\\Vodena_Primeri\\ImplementingEncodingInBf\\data\\test_model_encoding_version_bf_feature_selection_f.h5'
metadata = bf.get_ann_metadata(model_name)
print(metadata)

# Prepare input data for prediction
X_test_prepared_with_BF = prepare_input_data(X_test, metadata)





# Load model
from tensorflow.keras.models import load_model
model_BF_ann = load_model(model_name)

# Prediction and scale predicted data
y_pred_BF_ann = model_BF_ann.predict(X_test_prepared_with_BF)
y_pred_BF_ann = scale_output_data(y_pred_BF_ann, metadata)
print(y_test)





