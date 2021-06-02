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
    def get_available_decoder_types():
        return ["OneHot", "Dummy"]

    @staticmethod
    def encode(variable, variable_encoder_info):
        """
            Encoding each variable of test set as it was done in the training set.

            Parameters
            ----------
            data : numpy.array
                Data as numpy array.
            encoding_info : dict
                Variable encoding info from training data.

            Returns
            -------
            numpy.array
                Encoded variable.
        """
        variable = np.array(variable).reshape(len(variable), 1)
        encoding_type = variable_encoder_info['type']
        if encoding_type in VariableEncoding.get_available_encoder_types():
            variable_str = []
            for elem in variable:
                if type(elem) is not str:
                    variable_str.append(str(elem[0]))
                else:
                    variable_str.append(elem)
            variable_str = np.array(variable_str).reshape(len(variable_str), 1)
            #variable_str.dtype = str

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
        else:
            raise Exception ("Encoding type " + str(variable_encoder_info['type']) + " is not suported.")
    
    

    @staticmethod
    def decode(variable, variable_encoder_info, threshold):
        """
            Decode variable in relation to the encode method used in the optimization process.

            Parameters
            ----------
            variable : numpy.array
                Data as numpy array.
            variable_encoder_info : dict
                Variable encoding info from training input data.
            variable_encoder_info : float
                The threshold that determines the binary classes.

            Returns
            -------
            numpy.array
                Decoded variable.
        """

        decoding_type = variable_encoder_info['type']

        if decoding_type in VariableEncoding.get_available_decoder_types():
            if decoding_type == "OneHot":
                predicted_var = np.array(variable)
                predicted_var =  np.where(predicted_var == predicted_var.max(axis=1).reshape(len(predicted_var),1), 1, 0)

                training_input_categories = np.array(variable_encoder_info['categories'])           
                enc = OneHotEncoder(categories = [training_input_categories])
                enc.fit(training_input_categories.reshape(len(training_input_categories), 1))
                decoded_variable = enc.inverse_transform(predicted_var)
                decoded_variable = np.array([elem[0] for elem in decoded_variable]).reshape(len(decoded_variable), 1)
            else:
                predicted_var = np.array(variable).reshape(len(variable), 1)
                predicted_var =  np.where(predicted_var > threshold, 1, 0)

                training_input_categories = np.array(variable_encoder_info['categories'])  
                enc = OneHotEncoder(categories = [training_input_categories], drop = 'first')
                enc.fit(training_input_categories.reshape(len(training_input_categories), 1))
                decoded_variable = enc.inverse_transform(predicted_var)
            try:
                decoded_variable = np.array([int(float(elem[0])) for elem in decoded_variable]).reshape(len(decoded_variable), 1)
                return decoded_variable
            except:
                return decoded_variable

        else:
            raise Exception ("Decoding type " + str(variable_encoder_info['type']) + " is not suported.")