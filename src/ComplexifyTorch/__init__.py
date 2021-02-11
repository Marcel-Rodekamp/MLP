from .angle import *

from pathlib import Path
import torch

# get the path of the installation
module_path = Path(__file__).resolve().parent.parent.parent

# load the registered library
torch.ops.load_library(module_path/"libcomplexifyTorch_cpp.so")

# Alias the given operations
# cmatmul(Tensor,Tensor) -> Tensor
#cmatmul = torch.ops.complexifyTorch_cpp.cmatmul
cmatmul = torch.ops.complexifyTorch_cpp.cmatmul

# cadd(Tensor,Tensor) -> Tensor
cadd = torch.ops.complexifyTorch_cpp.cadd
