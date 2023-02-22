from .input_extras import remove_not_used_inputs
from .scaler_extras import scale_input_data, scale_output_data
from .series_extras import pack_input_data_for_series
from .series_extras import pack_output_data_for_series
from .rnn_extras import pack_input_data_for_rnn
from .rnn_extras import pack_output_data_for_rnn
from .encode_data_set import encode_inputs, encode_output_data

def prepare_input_data(input_data, metadata):
    """
        Prepare the input for prediction or training using following steps:
            1. Removing insignificant columns,
            2. Encoding of input features,
            3. Packing data for series,
            4. Scaling (normalizing) values,
            5. Pack data for recurrent neural network.
        Parameters
        ----------
        input_data : numpy.array
            Input data as numpy array.
        metadata : dict
            Model metadata.

        Returns
        -------
        numpy.array
            Prepared input values.
        """
    used_inputs = remove_not_used_inputs(input_data, metadata)
    if 'input_encodings' in metadata:
        used_inputs = encode_inputs(used_inputs, metadata['input_encodings'])
    if metadata['has_rolling']:
        used_inputs = pack_input_data_for_series(used_inputs, metadata)
    used_inputs = scale_input_data(used_inputs, metadata)
    if metadata.get('recurrent_input_count', None) is not None and metadata.get('recurrent_output_count', None) is not None:
        used_inputs = pack_input_data_for_rnn(used_inputs, metadata)
    return used_inputs


def prepare_output_data(output_data, metadata):
    """
        Prepare the output for prediction or training using following steps:
            1. Encode output data,
            2. Packing data for series,
            3. Scale output data,
            4. Pack data for recurrent neural network.
        Parameters
        ----------
        output_data : numpy.array
            Output data as numpy array.
        metadata : dict
            Model metadata.

        Returns
        -------
        numpy.array
            Prepared output values.
        """
    if 'output_encoding' in metadata:
        output_data = encode_output_data(output_data, metadata)
    if metadata['has_rolling']:
        output_data = pack_output_data_for_series(output_data, metadata)
    output_data = scale_output_data(output_data, metadata)
    if metadata.get('recurrent_input_count', None) is not None and metadata.get('recurrent_output_count', None) is not None:
        output_data = pack_output_data_for_rnn(output_data, metadata)
    return output_data

def pack_output_data_for_comparison_with_predictions(output_data, metadata):
    """
        Prepare the output for comparison using following steps:
            1. Encode output data,
            2. Packing data for series or pack data for recurrent neural network.
        Parameters
        ----------
        output_data : numpy.array
            Output data as numpy array.
        metadata : dict
            Model metadata.

        Returns
        -------
        numpy.array
            Prepared output values.
        """
    if 'output_encodings' in metadata:
        output_data = encode_output_data(output_data, metadata)
    if metadata['has_rolling']:
        output_data = pack_output_data_for_series(output_data, metadata)
    if metadata.get('recurrent_input_count', None) is not None and metadata.get('recurrent_output_count', None) is not None:
        output_data = pack_output_data_for_rnn(output_data, metadata)
    return output_data