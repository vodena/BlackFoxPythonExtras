from .input_extras import remove_not_used_inputs
from .scaler_extras import scale_data_input, scale_data_output
from .series_extras import pack_input_data_for_series
from .series_extras import pack_output_data_for_series


def prepare_input_data(input_data, metadata):
    """
        Prepare the input for prediction with the following steps
            1. removing insignificant columns
            2. packing data for series
            3. scaling (normalizing) values

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
    return scale_data_input(used_inputs, metadata)
