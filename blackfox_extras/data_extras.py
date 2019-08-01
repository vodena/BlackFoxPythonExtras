from .input_extras import remove_not_used_inputs
from .scaler_extras import scale_data_input, scale_data_output
from .series_extras import pack_input_data_for_series
from .series_extras import pack_output_data_for_series


def prepare_input_data(input_data, metadata):
    used_inputs = remove_not_used_inputs(input_data, metadata)
    scaled_data = scale_data_input(used_inputs, metadata)
    if metadata['has_rolling']:
        return pack_input_data_for_series(scaled_data, metadata)
    else:
        return scaled_data


def prepare_output_data(output_data, metadata):
    scaled_data = scale_data_output(output_data, metadata)
    if metadata['has_rolling']:
        return pack_output_data_for_series(scaled_data, metadata)
    else:
        return scaled_data


def prepare_data(data, metadata):
    input_count = len(metadata['scaler_config']['input']['fit'][0])
    if 'is_input_used' in metadata:
        input_count = len(metadata['is_input_used'])
    input_data = data[:, 0:input_count]
    p_input = prepare_input_data(input_data, metadata)
    p_output = None
    if len(data[0]) - input_count > 0:
        output_data = data[:, input_count:]
        p_output = prepare_output_data(output_data, metadata)

    return p_input, p_output
