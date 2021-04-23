import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('C:\\Users\\VODENA06\\Desktop\\Jupyter\\BF\\BF_TESTIRANJE\\5_0_0\\Churn_Modelling.csv')

print(dataframe.columns)

# df_X = dataframe.iloc[:, 3:13]
y = dataframe.iloc[:, 13:14].values
print(y[:5,:])
y = np.array([3 if elem==0 else 5 for elem in y]).reshape(len(y), 1)
print(y[:5,:])

#df_X = pd.concat(
#    [
#     pd.get_dummies(df_X['Geography'], drop_first = True),
#     pd.get_dummies(df_X['Gender'], drop_first = True),
#     df_X.drop(['Geography', 'Gender'], axis=1),
#     ], axis=1)
#X = df_X.iloc[:, :].values
X = dataframe.iloc[:, 3:13].values
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




# Get model metadata
metadata = {'__version': 8, 'scaler_name': 'MinMaxScaler', 'scaler_config': {'input': {'feature_range': [-1, 1], 'fit': [[350.0, 0.0, 0.0, 0.0, 0, 18.0, 0.0, 0.0, 1.0, 0.0, 0.0, 11.58], [850.0, 1.0, 1.0, 1.0, 1, 85.0, 10.0, 238387.56, 4.0, 1.0, 1.0, 199992.48]], 'inverse_transform': False}, 'output': {'feature_range': [0, 1], 'fit': [[0.0], [1.0]], 'inverse_transform': True}}, 'is_scaler_integrated': False, 'has_rolling': False, 'binary_metric': 'ROC_AUC', 'metric_threshold': None, 'input_encodings': [{'type': 'None'}, {'type': 'OneHot', 'categories': ['0', '1', '2']}, {'type': 'OrderInteger', 'mapping': {'1': 0, '0': 1}}, {'type': 'None'}, {'type': 'None'}, {'type': 'None'}, {'type': 'None'}, {'type': 'None'}, {'type': 'None'}, {'type': 'None'}], 'output_encoding': [{'type': 'Dummy', 'categories': ['3', '5']}]}

# Prepare input data for prediction
from data_extras import prepare_input_data
X_test_prepared_with_BF = prepare_input_data(X_test, metadata)

y_pred = [0.2,0.3,0.6,0.9,0,1,0.9,0,0,0,0,0,0]
from encode_variable import VariableEncoding
y_decoded = VariableEncoding.decode(y_pred, metadata)

print('Done.')