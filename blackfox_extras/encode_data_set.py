import numpy as np
from .encode_variable import VariableEncoding

def encode_inputs(input_data_set, input_encoding_info):
    """
        Test set input data encoding. Use before prediction.

        Parameters
        ----------
        input_data_set : numpy.array
            Data as numpy array
        input_encoding_info : [dict]
            Variables encoding info from training input data

        Returns
        -------
        numpy.array
            Encoded input data
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

def encode_output_data(output_data_set, output_encoding_info):
    """
        Test set output data encoding.

        Parameters
        ----------
        output_data_set : numpy.array
            Test set output data as numpy array
        output_encoding_info : [dict]
            Output encoding info from training output data

        Returns
        -------
        numpy.array
            Test set encoded output data
    """
    output_data_set = np.array(output_data_set)
    
    output_data_set = VariableEncoding.encode(output_data_set, output_encoding_info)
    
    return output_data_set

def decode_output_data(output_data_set, metadata):
    """
        Test set output data decoding. Use after prediction.

        Parameters
        ----------
        output_data_set : numpy.array
            Predicted data as numpy array
        output_encoding_info : [dict]
            Output encoding info from training output data

        Returns
        -------
        numpy.array
            Decoded predicted output data
    """

    if 'output_encoding' in metadata:
        output_data_set = VariableEncoding.decode(output_data_set, metadata['output_encoding'][0])
    else:
        raise Exception ("The output variable has not been encoded so this method cannot be applied.") 

    return output_data_set