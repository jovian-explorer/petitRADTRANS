from __future__ import print_function

import os
import sys


__author__ = "Paul Molliere"
__copyright__ = "Copyright 2016-2018, Paul Molliere"
__maintainer__ = "Paul Molliere"
__email__ = "molliere@strw.leidenunivl.nl"
__status__ = "Development"
__version__ = "2.1.0"

# Link to the libs folders on Windows
extra_dll_dir = os.path.join(os.path.dirname(__file__), '.libs')

if sys.platform == 'win32' and os.path.isdir(extra_dll_dir):
    if sys.version_info >= (3, 8):
        os.add_dll_directory(extra_dll_dir)
    else:
        os.environ.setdefault('PATH', '')
        os.environ['PATH'] += os.pathsep + extra_dll_dir


from petitRADTRANS.radtrans import Radtrans
