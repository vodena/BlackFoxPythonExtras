from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from category_encoders.sum_coding import SumEncoder
import pandas as pd
import numpy as np

class VariableEncoding:
    @staticmethod
    def get_available_encoder_types():
        return ["None", "OneHot", "Dummy", "Target", "Effect", "CountOfFrequency", "OrderInteger"]

    @staticmethod
    def encode(variable, variable_encoder_info):
        """
            Encoding each variable of test set as it was done in the training set

            Parameters
            ----------
            data : numpy.array
                Data as numpy array
            input_encoding_info : dict
                Variable encoding info from training input data

            Returns
            -------
            numpy.array
                Encoded variable
        """
        encoding_type = variable_encoder_info['type']
        if encoding_type in VariableEncoding.get_available_encoder_types():
            variable_str = []
            for elem in variable:
                if type(elem) is not str:
                    variable_str.append(str(elem[0]))
                else:
                    variable_str.append(elem)
            variable_str = np.array(variable_str).reshape(len(variable_str), 1)

            if encoding_type == "OneHot":
                training_input_categories = np.array(variable_encoder_info['categories'])           
                enc = OneHotEncoder(categories = [training_input_categories])
                enc.fit(training_input_categories.reshape(len(training_input_categories), 1))
                encoded_variable = enc.transform(variable_str).toarray()
            elif encoding_type == "Dummy":
                training_input_categories = np.array(variable_encoder_info['categories'])           
                enc = OneHotEncoder(categories = [training_input_categories], drop = 'first')
                enc.fit(training_input_categories.reshape(len(training_input_categories), 1))
                encoded_variable = enc.transform(variable_str).toarray()
            elif encoding_type == "Target":
                training_input_mapping = variable_encoder_info['mapping']
                variable_str = pd.Series(variable_str.reshape(len(variable_str),))
                encoded_variable = variable_str.map(training_input_mapping)
                encoded_variable = np.array(encoded_variable).reshape(len(encoded_variable), 1)
            elif encoding_type == "Effect":
                variable_str = list(variable_str.reshape(len(variable_str),))
                encoded_variable = np.array([variable_encoder_info['mapping'][elem] for elem in variable_str])
            elif encoding_type == "CountOfFrequency":
                variable_str = pd.Series(variable_str.reshape(len(variable_str), ))
                encoded_variable = variable_str.map(variable_encoder_info['mapping'])
                encoded_variable = np.array(encoded_variable).reshape(len(encoded_variable), 1)
            else:
                variable_str = pd.Series(variable_str.reshape(len(variable_str), ))
                encoded_variable = variable_str.map(variable_encoder_info['mapping'])
                encoded_variable = np.array(encoded_variable).reshape(len(encoded_variable), 1)
                
            return encoded_variable