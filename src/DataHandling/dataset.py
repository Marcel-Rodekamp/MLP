import h5py as h5
import torch

def loadHDF5Data(fn, dataSet_path_flowed="phi_flowed", dataSet_path_unflowed="phi_unflowed",stride=None):
    r"""!
        fn: string
            Path to the HDF5 Data
        dataSet_path: string, default: "phi_flowed"
            Metadata path of the HDF5 file to the configuration
            Defaults to a flowed configuration used as input to the training of
            the CVN.

        Returns: torch.tensor
            All

        Notes:
            Loads data from a HDF5 file.

        Assumes:
            * Data is complex valued double precision

    """

    import h5py as h5
    import torch
    import numpy as np

    with h5.File(fn,'r') as f:
        if stride is None:
            confs_flowed = torch.from_numpy(f[dataSet_path_flowed][()].astype( np.cdouble ))
            confs_unflowed = torch.from_numpy(f[dataSet_path_unflowed][()].astype( np.cdouble ))
        else:
            confs_flowed = torch.from_numpy(f[dataSet_path_flowed][()].astype( np.cdouble )[::10])
            confs_unflowed = torch.from_numpy(f[dataSet_path_unflowed][()].astype( np.cdouble )[::10])

    return confs_unflowed,confs_flowed

# Implement the data set class to interface to the data management of pytorch
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,t_fn,on_gpu_flag = False, device = torch.device("cpu"),dataSet_path_flowed="phi_flowed", dataSet_path_unflowed="phi_unflowed"):
        'Initialization'
        self.unflowed_confs,self.flowed_confs = loadHDF5Data(t_fn,dataSet_path_flowed,dataSet_path_unflowed)
        if on_gpu_flag:
            self.to(device)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.flowed_confs)

    def to(self,*args,**kwargs):
        self.unflowed_confs = self.unflowed_confs.to(*args,**kwargs)
        self.flowed_confs = self.flowed_confs.to(*args,**kwargs)

    def cpu(self,memory_format=torch.preserve_format):
        self.unflowed_confs = self.unflowed_confs.cpu(memory_format)
        self.flowed_confs = self.flowed_confs.cpu(memory_format)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.unflowed_confs[index][:], self.flowed_confs[index][:]
