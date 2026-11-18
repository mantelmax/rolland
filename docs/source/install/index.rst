Installation
============

Prerequisites
-------------
**Rolland** requires **Python 3.12** or newer. 

We strongly recommend installing Rolland inside a virtual environment (e.g., using ``venv`` or ``conda``) to avoid conflicts with other packages on your system.

Using pip
---------
The easiest way to install Rolland is via ``pip``. Run the following command in your terminal:

.. code-block:: bash

   pip install rolland

To update Rolland to the latest version, run:

.. code-block:: bash

   pip install --upgrade rolland

*(Optional: To install a specific version, use `pip install rolland==<version>`)*

Platform Support & Windows
--------------------------
Rolland is fully supported and tested on **Linux** and **macOS**.

**Windows Users:** 
Currently, Rolland works on Windows, but it requires a workaround due to compatibility issues with the underlying `Devito <https://github.com/devitocodes/devito>`_ dependency. 
We highly recommend using **WSL (Windows Subsystem for Linux)** for a seamless experience. If you must run it natively on Windows, please refer to the official `Devito Installation Issues <https://github.com/devitocodes/devito/wiki/Installation-Issues>`_ guide for the required workaround.

Verifying the Installation
--------------------------
After installing, you can verify that Rolland is correctly installed by checking its version:

.. code-block:: bash

   python -c "import rolland; print(rolland.__version__)"   

This should print the currently installed version of Rolland.

Installing from Source
----------------------
If you want to contribute to the development or use the latest unreleased features, you can install Rolland directly from the source repository:

.. code-block:: bash

   git clone https://github.com/mantelmax/rolland.git
   cd rolland
   pip install -e .
