import numpy as np
from .encodeVariable import VariableEncoding

def encode(input_data_set, input_encoding_info):
    """
        Encoding input data for test set

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        input_encoding_info : [dict]
            Variables encoding info from training input data

        Returns
        -------
        numpy.array
            Encoded data
    """
    input_data_set = np.array(input_data_set)

    i = 0
    enc_index = 0
    number_of_variables = len(input_encoding_info)
    while i < number_of_variables:
        variable_encoder_info = input_encoding_info[enc_index]

        if variable_encoder_info['type'] != 'None':
            variable = input_data_set[:, i:i+1]
            encoded_variable = VariableEncoding.encode(variable, variable_encoder_info) # encode each variable as in training
            
            if i == 0:
                input_data_set = np.concatenate((encoded_variable, input_data_set[:, i+1:]), axis=1)
            elif i == number_of_variables-1:
                input_data_set = np.concatenate((input_data_set[:, :i], encoded_variable), axis=1)
            else:
                input_data_set = np.concatenate((input_data_set[:, :i], encoded_variable, input_data_set[:, i+1:]), axis=1)

            if variable.shape[1] != encoded_variable.shape[1]:
                number_of_variables = number_of_variables - variable.shape[1] + encoded_variable.shape[1]
                i = i - variable.shape[1] + encoded_variable.shape[1]
        i += 1
        enc_index += 1
    
    return input_data_set