Install
========

PIP
***

The package can be cloned from CNPEM's gitlab and hence install locally with your user:

.. code-block:: bash

   git clone https://gitlab.cnpem.br/GCC/ssc-pimega.git

   python3 setup.py install --user --cuda


.. note:: Flag --cuda is optional for users with a Nvidia/GPU.

REQUIREMENTS
************

For completeness, the following packages are used within this package:

.. code-block:: python 

        import os
        import sys
        import ctypes
	import numpy
	import time
	import gc
	import h5py
	
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	import glob
	
	import multiprocessing
	import multiprocessing.sharedctypes
	
	from threading import Thread
	
	import functools
	from functools import partial
	
	from PIL import Image
	import gc
	import subprocess
	import SharedArray as sa
	import uuid
	
	from scipy.optimize import minimize
	
	from scipy import ndimage	
	import pickle


MEMORY
******

Except for CUDA functions, which are able to restore a block of images, this package does not require a GPU.
