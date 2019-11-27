import numpy as np

def pack_input_data_for_rnn(input_data, metadata):
    """
        Packing input data for recurrent neural network

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        metadata : dict
            Model metadata

        Returns
        -------
        numpy.array
            Packed data
    """
    rnn_input_count = metadata['recurrent_input_count']
    rnn_output_count = metadata['recurrent_output_count']

    x_data = []
    rows_count = input_data.shape[0] - rnn_output_count
    for i in range(rnn_input_count, rows_count):
        x_data.append(input_data[i - rnn_input_count:i])

    new_shape = (rows_count - rnn_input_count, rnn_input_count, input_data.shape[1])
    x_data = np.reshape(x_data, new_shape)
    return x_data

def pack_output_data_for_rnn(output_data, metadata):
    """
        Packing output data for recurrent neural network

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        metadata : dict
            Model metadata

        Returns
        -------
        numpy.array
            Packed data
    """
    rnn_input_count = metadata['recurrent_input_count']
    rnn_output_count = metadata['recurrent_output_count']

    y_data = []
    rows_count = output_data.shape[0] - rnn_output_count
    for i in range(rnn_input_count, rows_count):
        y_data.append(output_data[i:i + rnn_output_count].flatten())

    y_data = np.array(y_data)
    return y_data