# from .encodeDataSet import encode

# import pandas as pd
# from blackfox import BlackFox

# # blackfox_url = 'http://147.91.204.14:32706'
# blackfox_url = 'http://localhost:5000'

# bf = BlackFox(blackfox_url)
# metadata = bf.get_ann_metadata('F:\\.spyder-py3\\projekti\\Vodena_Primeri\\ImplementingEncodingInBf\\data\\11111111-1111-1111-1111-111111111111\\test_model_encoding_version_bf.h5')
# print(metadata)


# dataframe = pd.read_csv('F:\\.spyder-py3\\projekti\\Vodena_Primeri\\ImplementingEncodingInBf\\data\\data_set.csv')
# used_inputs = dataframe.iloc[:, :-1].values   

# if 'input_encodings' in metadata:
#         used_inputs = encode(used_inputs, metadata['input_encodings'])
# else:
#     print('nema')
        
        

# Importing Black Fox service libraries

from blackfox import BlackFox
from blackfox import AnnOptimizationConfig, RandomForestOptimizationConfig, XGBoostOptimizationConfig
from blackfox import AnnOptimizationEngineConfig, OptimizationEngineConfig
from blackfox import LogWriter, CsvLogWriter
from blackfox_extras import prepare_input_data, scale_output_data

# blackfox_url = 'http://147.91.204.14:32702' # BF_URL
# blackfox_url = 'http://147.91.204.85:32723'# ~VODAFONE_URL
# blackfox_url = 'http://147.91.204.14:32706' # BF FOR MAPE verziju blackfox 3.2.0
blackfox_url = 'http://localhost:5000'
bf = BlackFox(blackfox_url)

# Importing libraries

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

# Import data

dataframe = pd.read_csv('F:\\.spyder-py3\\projekti\\Vodena_Primeri\\ANN\\Churn\\Churn_Modelling.csv')

print(dataframe.columns)


# Importing libraries

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

# Import data

# dataframe = pd.read_csv('Churn_Modelling.csv')

# Data preparation

# df_X = dataframe.iloc[:, 3:13]
y = dataframe.iloc[:, -1:].values

#df_X = pd.concat(
#    [
#     pd.get_dummies(df_X['Geography'], drop_first = True),
#     pd.get_dummies(df_X['Gender'], drop_first = True),
#     df_X.drop(['Geography', 'Gender'], axis=1),
#     ], axis=1)
#X = df_X.iloc[:, :].values
# X = dataframe.iloc[:, :-1].values

X = dataframe.iloc[:, 3:-1].values
print(X[:3,:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X = np.concatenate(
    [
    X[:, 0:1],
    le.fit_transform(X[:, 1:2]).reshape(len(X),1),
    le.fit_transform(X[:, 2:3]).reshape(len(X),1),
    X[:, 3:]
    ], axis=1)
print(X[:3,:])

# Split the entire data set into the training data and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('shape 0 :{}      and shape 1 :{}'.format(X_train.shape[0], X_train.shape[1]))
print('shape 0 :{}      and shape 1 :{}'.format(X_test.shape[0], X_test.shape[1]))
print('shape 0 :{}      and shape 1 :{}'.format(y_train.shape[0], y_train.shape[1]))
print('shape 0 :{}      and shape 1 :{}\n'.format(y_test.shape[0], y_test.shape[1]))

# Split the training dada into the training set and validation set

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
print('shape 0 :{}      and shape 1 :{}'.format(X_train.shape[0], X_train.shape[1]))
print('shape 0 :{}      and shape 1 :{}'.format(X_val.shape[0], X_val.shape[1]))
print('shape 0 :{}      and shape 1 :{}'.format(y_train.shape[0], y_train.shape[1]))
print('shape 0 :{}      and shape 1 :{}\n'.format(y_val.shape[0], y_val.shape[1]))

from blackfox import InputConfig, Encoding

start = time.time()

# ec = AnnOptimizationEngineConfig(
#     population_size = 20,
#     max_num_of_generations = 20
# )

# c = AnnOptimizationConfig(
#     problem_type = "BinaryClassification",
#     binary_optimization_metric = "ROC_AUC",
#     engine_config = ec,
#     inputs=[
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.DUMMY),
#         InputConfig(encoding='Dummy'),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE),
#         InputConfig(encoding=Encoding.NONE)
#     ]
#     )

# # Use CTRL + C to stop optimization
# (ann_io, ann_info, ann_metadata) = bf.optimize_ann(
#     input_set = X_train,
#     output_set = y_train,
#     input_validation_set = X_val,
#     output_validation_set = y_val,
#     config = c,
#     model_path = 'bf_model_churn_testiranje_bf_3_3_0_ENCODED.h5',
#     delete_on_finish = False
# )

# end = time.time()
# time_BF_ann = int(end-start)

# print('\nann info:')
# print(ann_info)

# print('\nann metadata:')
# print(ann_metadata)

# Get model metadata
model_name = 'C:\\Users\\VODENA06\\Desktop\\Jupyter\\BF\\BF_TESTIRANJE\\bf_model_churn_testiranje_bf_3_3_0_ENCODED.h5'
metadata = bf.get_ann_metadata(model_name)
print(metadata)

# Prepare input data for prediction
X_test_prepared_with_BF = prepare_input_data(X_test, metadata)



