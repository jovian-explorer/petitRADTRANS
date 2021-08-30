Installation
============

Pre-installation: download the opacity data
___________________________________________

Before you install pRT, please download the opacity data, at least the
low-resolution version (:math:`\lambda/\Delta\lambda=1000`), as it
provides all relevant input files for pRT to run, and contains the
necessary folder structure if you want to install high-resolution
opacities later (:math:`\lambda/\Delta\lambda=10^6`).

Thus, to get started download the `opacity and input data
<https://keeper.mpdl.mpg.de/f/f5aba635d3a244adb3c0/?dl=1>`_
(12.1 GB), unzip them, and put the "input_data" folder somewhere on
your computer (it does not matter where).

Next, please add the following environment variable to your
“.bash_profile” or “.bashrc” file (depending on your operating system)
by typing 

.. code-block:: bash

   echo 'export pRT_input_data_path="absolute/path/of/the/folder/input_data"' >>~/.bash_profile

for Mac OS and

.. code-block:: bash

   echo 'export pRT_input_data_path="absolute/path/of/the/folder/input_data"' >>~/.bashrc

for Linux. Now you are ready to go and can proceed with the actual
installation of pRT.

.. attention::
   Don’t forget to adapt the path in the line above! If you are
   uncertain what the absolute path of the input_data folder is, then
   switch to that folder in the terminal, type “pwd”, and press Enter.
   You can then just copy-paste that path. Then close and reopen the
   terminal such that it will read the environment variable correctly.

If you want to also use high-resolution opacity
data please follow these steps here, but note that they can be
installed at any point after the pRT installation:

The high resolution (:math:`\lambda/\Delta\lambda=10^6`) opacity data
(about 240 GB if you want to get all species) can be
accessed and downloaded `via Keeper here`_. To
install them, create a folder called "line_by_line" in the
"input_data/opacities/lines" folder. Then put the folder of the absorber
species you downloaded in there.

.. _`via Keeper here`: https://keeper.mpdl.mpg.de/d/e627411309ba4597a343/

Installation via pip install
____________________________

To install pRT via pip install just type

.. code-block:: bash

   pip install petitRADTRANS

in a terminal. Note that you must also have downloaded the low-resolution
opacities either before or after to actually run pRT, see
`above <#pre-installation-download-the-opacity-data>`_.

Compiling pRT from source
_________________________

Download petitRADTRANS from `Gitlab <https://gitlab.com/mauricemolli/petitRADTRANS.git>`_, or clone it from GitLab via

.. code-block:: bash
		
   git clone git@gitlab.com:mauricemolli/petitRADTRANS.git

- In the terminal, enter the petitRADTRANS folder
- Type the following in the terminal ``python setup.py install``, and press
  Enter.

Testing the installation
________________________

Open a new terminal window (this will source the ``pRT_input_data_path``). Then open python and type

.. code-block:: python
		
   from petitRADTRANS import Radtrans
   atmosphere = Radtrans(line_species = ['CH4'])

This should produce the following output:

.. code-block:: bash
		
     Read line opacities of CH4...
    Done.
