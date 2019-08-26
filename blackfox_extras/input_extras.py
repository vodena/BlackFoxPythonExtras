

def remove_not_used_inputs(data, metadata):
    """
        Remove insignificant columns from data, if model has feature selection

        Parameters
        ----------
        data : numpy.array
            Data as numpy array
        metadata : dict
            Model metadata

        Returns
        -------
        numpy.array
            New data
        """
    if 'is_input_used' in metadata:
        tuples = [[x, i] for i, x in enumerate(metadata['is_input_used'])]
        used = filter(lambda tu: tu[0], tuples)
        index = list(map(lambda t: t[1], used))
        return data[:, index]
    else:
        return data
