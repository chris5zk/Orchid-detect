import torch
import sys

from bcolors import *

print(bcolors.OKBLUE + "cuda" + bcolors.ENDC if torch.cuda.is_available() else bcolors.WARNING + "cpu" + bcolors.ENDC, file=sys.stderr)
