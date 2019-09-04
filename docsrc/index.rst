.. blackfox-extras documentation master file, created by
   sphinx-quickstart on Tue Sep  3 13:19:36 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Black Fox extras documentation
==============================

Black Fox extras is the companion package for the `Black Fox <http://blackfox.ai/>`_, *the neural network parameter optimization AI automation assistant*.

Black Fox extras offer performant preparatory functions for both input and output, as well as for time series data.
These functions are integrated into the Black Fox workflow, and they facilitate tedious tasks like removing the insignificant
variables (columns), data scaling, and transformations needed for time series data.

.. toctree::
    :maxdepth: 2

User's Guide
------------

Installation
~~~~~~~~~~~~
To install Black Fox extras use `pip <https://pip.pypa.io/en/stable/quickstart/>`_ or `pipenv <https://docs.pipenv.org/en/latest/>`_:

.. code-block:: PowerShell

    $ pip install -U blackfox-extras

Example usage
~~~~~~~~~~~~~

Model prediction with preparing data and calculating mean absolute error:

.. code-block:: python

    from blackfox import BlackFox
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from keras.models import load_model

    # Create an instance of the Black Fox class by supplying api URL
    bf = BlackFox('bf.endpoint.api.address')

    ann_file = 'model.h5'

    # Get model metadata and load model
    ann_metadata = bf.get_metadata(ann_file)
    model = load_model(ann_file)

    # Read data
    data = pd.read_csv('data.csv')

    # Get all columns except last as input values and last columns as output
    x_data = data.iloc[:,:-1].values
    y_data = data.iloc[:,-1:].values

    # Pack output data
    y_real = pack_output_data_for_series(y_data, ann_metadata)

    # Prepare input data for prediction
    x_prepared = prepare_input_data(x_data, ann_metadata)
    
    # Prediction and scale predicted data
    y_predicted = model.predict(x_prepared)
    y_predicted = scale_data_output(y_predicted, ann_metadata)

    # Calculate MAE
    mae = mean_absolute_error(y_predicted, y_real)

API Guide
---------
Module
~~~~~~~~~~~~~~~
.. automodule:: blackfox_extras
    :members:

.. autofunction:: prepare_input_data
.. autofunction:: pack_output_data_for_series
.. autofunction:: scale_output_data

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
