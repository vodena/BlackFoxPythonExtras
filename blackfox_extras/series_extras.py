import numpy as np


def get_series_padding(metadata):
    input_count = len(metadata['input_shifts'])
    output_count = len(metadata['output_shifts'])

    in_range = range(input_count)
    out_range = range(output_count)

    left_input = max(list(map(
        lambda x: metadata['input_shifts'][x] + metadata['input_windows'][x] - 1, in_range)))
    left_output = max(list(map(
        lambda x: metadata['output_shifts'][x] + metadata['output_windows'][x] - 1, out_range)))

    right_input = min(
        list(map(lambda x: metadata['input_shifts'][x], in_range)))
    right_output = min(
        list(map(lambda x: metadata['output_shifts'][x], out_range)))

    sample_step = metadata['output_sample_step']
    left_padding = max(0, left_input, left_output)
    left_padding = int(sample_step * np.ceil(left_padding / sample_step))
    right_padding = min(right_input, right_output, 0)

    return left_padding, right_padding


def pack_input_data_for_series(input_data, metadata):
    input_count = len(metadata['input_shifts'])
    sample_step = metadata['output_sample_step']

    left_padding, right_padding = get_series_padding(metadata)

    x_data = []
    rows_count = input_data.shape[0]+right_padding
    for i in range(left_padding, rows_count, sample_step):
        x_row = __get_input_row(input_data, metadata, i, input_count)
        x_data.append(x_row)

    x_data = np.array(x_data)
    return x_data


def pack_output_data_for_series(output_data, metadata):
    output_count = len(metadata['output_shifts'])
    sample_step = metadata['output_sample_step']

    left_padding, right_padding = get_series_padding(metadata)

    y_data = []
    rows_count = output_data.shape[0]+right_padding
    for i in range(left_padding, rows_count, sample_step):
        y_row = __get_output_row(output_data, metadata, i, output_count, 0)
        y_data.append(y_row)

    y_data = np.array(y_data)
    return y_data


def pack_data_for_series(data, metadata):
    input_count = len(metadata['input_shifts'])
    output_count = len(metadata['output_shifts'])
    sample_step = metadata['output_sample_step']

    left_padding, right_padding = get_series_padding(metadata)

    x_data = []
    y_data = []
    rows_count = data.shape[0]+right_padding
    for i in range(left_padding, rows_count, sample_step):
        
        x_row = __get_input_row(data, metadata, i, input_count)
        x_data.append(x_row)

        y_row = __get_output_row(data, metadata, i, output_count, input_count)
        y_data.append(y_row)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return (x_data, y_data)


def __get_input_row(data, metadata, i, input_count):
    x_row = []
    for j in range(input_count):
        c = data.columns[j]
        shift = metadata['input_shifts'][j]
        window = metadata['input_windows'][j]
        step = metadata['input_steps'][j]
        aggregation_type = metadata['input_aggregation_types'][j]
        if step is None:
            step = window if aggregation_type != 'none' else 1
        offset = i+1-shift
        d = data[c][offset-window:offset]
        n = len(d)
        if aggregation_type == 'sum':
            r = range((n) % step, n-1, step)
            new_d = [sum(d[s:s+step]) for s in r]
        elif aggregation_type == 'avg':
            r = range((n) % step, n-1, step)
            new_d = [np.mean(d[s:s+step]) for s in r]
        else:
            r = range((n-1) % step, n, step)
            new_d = [d.values[s] for s in r]

        x_row = np.concatenate([x_row, new_d])

    return x_row


def __get_output_row(data, metadata, i, output_count, input_offset=0):
    y_row = []
    for j in range(output_count):
        c = data.columns[input_offset+j]
        shift = metadata['output_shifts'][j]
        window = metadata['output_windows'][j]
        offset = i+1-shift
        d = data[c][offset-window:offset]
        y_row = np.concatenate([y_row, d])

    return y_row
