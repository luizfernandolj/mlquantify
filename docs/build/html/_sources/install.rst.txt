.. _installation:

====================
Installing mlquantify
====================

The `mlquantify` library can be easily installed via `pip`. Below are detailed installation instructions, including its dependencies and differences in using a virtual environment across operating systems.


Dependencies
------------
Before installing `mlquantify`, ensure you have the following dependencies:

- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`
- `tqdm`
- `matplotlib`
- `xlrd`

`pip` will automatically install these dependencies if they are not already present in your environment.

Installation
------------
To install the library, use the following command:

.. code-block:: bash

    pip install mlquantify


Using Virtual Environments
----------------------------
It is recommended to use virtual environments to manage project dependencies. Below are the instructions for different operating systems.

### Linux/macOS

1. Create a virtual environment:


   .. code-block:: bash

       python3 -m venv mlquantify_env

2. Activate the virtual environment:

   .. code-block:: bash

       source mlquantify_env/bin/activate

3. Install the library:

   .. code-block:: bash

       pip install mlquantify

4. To deactivate the virtual environment:

   .. code-block:: bash

       deactivate


### Windows

1. Create a virtual environment:

   .. code-block:: powershell

       python -m venv mlquantify_env

2. Activate the virtual environment:

   .. code-block:: powershell

       mlquantify_env\Scripts\activate

3. Install the library:

   .. code-block:: powershell

       pip install mlquantify

4. To deactivate the virtual environment:

   .. code-block:: powershell

       deactivate

Now, `mlquantify` is ready to be used in your project!
