Installation
========================
To install **Rolland**, run in your terminal with the appropriate Python environment activated:

.. code-block:: bash

   pip install rolland

.. note::

   **Platform Support:**
   
   Rolland is fully supported and works on **Linux** and **macOS**.

   Currently, Rolland does work on **Windows**, but only with a workaround due to a compatibility conflict with the `Devito <https://github.com/devitocodes/devito>`_ dependency. For details and instructions on the required workaround, please refer to the official `Devito Installation Issues <https://github.com/devitocodes/devito/wiki/Installation-Issues>`_ guide (or consider using Windows Subsystem for Linux - WSL).

To update **Rolland** to the latest version, run:

.. code-block:: bash

   pip install --upgrade rolland

After installing **Rolland**, you can verify the installation by running:

..code-block:: bash

   python -c "import rolland; print(rolland.__version__)"   

This should print the installed version of **Rolland**.

To install a specific version of **Rolland**, run:

.. code-block:: bash

   pip install rolland==<version>

