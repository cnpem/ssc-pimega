try:
    import pkg_resources 
    __version__ = pkg_resources.require("sscPimega")[0].version
except:
    pass


from .pimegatypes import *
from .pi135D import *
from .pi540D import *
from .pi450D import *
from .optimize import *
from .misc import *

if __name__ == "__main__":
   pass


