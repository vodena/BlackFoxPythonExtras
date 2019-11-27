from .input_extras import remove_not_used_inputs
from .scaler_extras import scale_input_data, scale_output_data
from .series_extras import pack_input_data_for_series
from .series_extras import pack_output_data_for_series
from .rnn_extras import pack_input_data_for_rnn
from .rnn_extras import pack_output_data_for_rnn


def prepare_input_data(input_data, metadata):
    """
        Prepare the input for prediction with the following steps
            1. removing insignificant columns
            2. packing data for series
            3. scaling (normalizing) values
            4. pack data for recurrent neural network
        Parameters
        ----------
        input_data : numpy.array
            Input data as numpy array
        metadata : dict
            Model metadata

        Returns
        -------
        numpy.array
            Prepared values
        """
    used_inputs = remove_not_used_inputs(input_data, metadata)
    if metadata['has_rolling']:
        used_inputs = pack_input_data_for_series(used_inputs, metadata)
    used_inputs = scale_input_data(used_inputs, metadata)
    if metadata.get('recurrent_input_count', None) is not None and metadata.get('recurrent_output_count', None) is not None:
        used_inputs = pack_input_data_for_rnn(used_inputs, metadata)
    return used_inputs


def prepare_output_data(output_data, metadata):
    """
        Prepare the output for prediction with the following steps
            1. packing data for series or pack data for recurrent neural network
        Parameters
        ----------
        output_data : numpy.array
            Output data as numpy array
        metadata : dict
            Model metadata

        Returns
        -------
        numpy.array
            Prepared values
        """
    if metadata['has_rolling']:
        output_data = pack_output_data_for_series(output_data, metadata)
    if metadata.get('recurrent_input_count', None) is not None and metadata.get('recurrent_output_count', None) is not None:
        output_data = pack_output_data_for_rnn(output_data, metadata)
    return output_data