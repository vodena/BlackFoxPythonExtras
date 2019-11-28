from sklearn.preprocessing import MinMaxScaler
import copy

def scale_input_data(input_data, metadata, ignore_integrated_scaler=False):
    """
        Scale input data from real values to normalized.

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        metadata : dict
            Model metadata
        ignore_integrated_scaler: bool
            If False(default), only scale data if model does not contain integrated scaler

        Returns
        -------
        numpy.array
            Scaled data
        """
    if ignore_integrated_scaler or not metadata['is_scaler_integrated']:
        return __scale_data_from_config(
            input_data,
            metadata['scaler_config']['input'],
            metadata['scaler_name']
        )
    else:
        return input_data



def scale_output_data(output_data, metadata, ignore_integrated_scaler=False):
    """
        Scale data from normalized values to real values. Use after prediction.

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        metadata : dict
            Model metadata
        ignore_integrated_scaler: bool
            If False(default), only scale data if model does not contain integrated scaler

        Returns
        -------
        numpy.array
            Scaled data
        """
    if ignore_integrated_scaler or not metadata['is_scaler_integrated']:
        recurrent_output_count = metadata.get('recurrent_output_count', None)
        config = metadata['scaler_config']['output']
        if recurrent_output_count is not None:
            config = copy.deepcopy(config)
            config['fit'][0] = config['fit'][0] * recurrent_output_count
            config['fit'][1] = config['fit'][1] * recurrent_output_count
        return __scale_data_from_config(
            output_data,
            config,
            metadata['scaler_name']
        )
    else:
        return output_data


def __scale_data_from_config(data, config, scaler_name='MinMaxScaler'):
    if scaler_name == 'MinMaxScaler':
        return __min_max_scale_data(
            data,
            config['fit'],
            config['feature_range'],
            config['inverse_transform']
        )
    else:
        raise Exception('Unknown scaler ' + scaler_name)


def __min_max_scale_data(data, fit, feature_range, inverse=False):
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(fit)
    if inverse is True:
        return scaler.inverse_transform(data)
    else:
        return scaler.transform(data)
