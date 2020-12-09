import sys
from collections.abc import Iterable

import gt4py
import gt4py.storage as gt_storage
import netCDF4 as nc
import numpy as np
import time
from gt4py.gtscript import I, J, K, IJ, IJK, Field, stencil

from riem_solver_c import riem_solver_c
num_iter = 1
def iterable(obj):
    return isinstance(obj, Iterable)

class Dataset:
    def __init__(self, file_name, backend):
        self._dataset = nc.Dataset(file_name)
        self._backend = backend
        self.npx = int(self._dataset["npx"][0].item())
        self.npy = int(self._dataset["npy"][0].item())
        self.ng = int(self._dataset["ng"][0].item())
        self.km = int(self._dataset["km"][0].item()) + 1
        self.full_domain_nx = 2 * self.ng + self.npx - 1
        self.full_domain_ny = 2 * self.ng + self.npx - 1
    @staticmethod
    def _find_fuzzy(axes, name):
        return next(i for i, axis in enumerate(axes) if axis.startswith(name))

    def netcdf_to_gt4py(self, var):
        """Convert a netcdf variable to gt4py storage."""
        axes = [d.name for d in var.get_dims()]
        idim = self._find_fuzzy(axes, "xaxis")
        jdim = self._find_fuzzy(axes, "yaxis")
        kdim = self._find_fuzzy(axes, "zaxis")
        origin= (self.ng, self.ng, 0)
        if np.prod(var.shape) > 1:
            permutation = [dim for dim in (idim, jdim, kdim) if dim]
            # put other axes at the back
            for i in range(len(axes)):
                if i not in permutation:
                    permutation.append(i)
            ndarray = np.squeeze(np.transpose(var, permutation))
            if len(ndarray.shape) == 3:
                #origin = (self.ng, self.ng, 0)
                if ndarray.shape[2] < self.km:
                    newarr = np.zeros((self.full_domain_nx, self.full_domain_ny, self.km))
                    newarr[:, :, :ndarray.shape[2] - self.km] = ndarray
                    ndarray = newarr
            elif len(ndarray.shape) == 2:
                #origin = (self.ng, self.ng)
                ndarray = np.repeat(ndarray[:, :, np.newaxis], self.km, axis=2)
            else:
                origin = (0,)
            #print('FIELD', var.name, ndarray.shape)
            #if var.name == "pef":
            #    ndarray[:] = 1.0e8
            return gt_storage.from_array(
                ndarray, backend, default_origin=origin, shape=ndarray.shape)
        else:
            if var.name in ["q_con", "cappa"]:
                newarr = np.zeros((self.full_domain_nx, self.full_domain_ny, self.km))
                #newarr[:] = var[0].item()
                return gt_storage.from_array(newarr, backend, default_origin=(self.ng, self.ng, 0), shape=newarr.shape)
            else:
                return var[0].item()

    def __getitem__(self, index):
        variable = self._dataset[index]
        return self.netcdf_to_gt4py(variable)

    def new(self, axes, dtype, pad_k=False):
        k_add = 0 if pad_k else -1
        if axes == IJK:
            origin = (self.ng, self.ng, 0)
            shape = (self.full_domain_nx, self.full_domain_ny, self.km + k_add)
            mask = None
        elif axes == IJ:
            mask = (True, True, False)
            origin = (self.ng, self.ng)
            shape = (self.full_domain_nx, self.full_domain_ny)
        elif axes == K:
            mask = (False, False, True)
            origin = (0,)
            shape = (self.km + k_add, )
        else:
            raise ValueError("Axes unrecognized")
        return gt_storage.empty(backend=self._backend, default_origin=origin, shape=shape, dtype=dtype, mask=mask)


def do_test(data_file, backend):
    data = Dataset(data_file, backend)

    # other fields
    pe = data.new(IJK, float, pad_k=True)
    pef = data.new(IJK, float, pad_k=True)
    pem = data.new(IJK, float, pad_k=True)
    dm = data.new(IJK, float, pad_k=True)
    pef.data[:] = 1e8
    gz_old = np.zeros(pe.shape)
    gz_old = np.copy(data["gz"].data)
    print('gz start', data["gz"][:,2,0])
    print('gz ref', data["gz_out"][:,2,0])
    gz_old = gt_storage.from_array(gz_old, backend, default_origin=(data.ng, data.ng, 0), shape=gz_old.shape)
    riem = stencil(backend=backend, rebuild=False, definition=riem_solver_c, externals={"A_IMP": data["a_imp"]})
    #for var in ["hs", "w3", "pt", "q_con", "delp", "gz", "pef", "ws", "p_fac", "scale_m", "ms", "dt", "akap", "cp", "ptop", "cappa"]:
    #    print(var, type(data[var]))
    # saved as a singleton but should be 3d, probably due to unspecified dimension endpoints:
    start_time = time.time()
    gama = 1.0 / (1.0 - data["akap"])
    print('gz start', data["gz"][3, 3, 0:5], data["gz"][3, 3, 1:6], "dz2", data["gz"][3, 3, 1:6] - data["gz"][3, 3, 0:5])
    print('ws', data["ws"][3, 3, 0:7])
    for iter in range(num_iter):
        riem( pem, dm, data["hs"],
              data["w3"],
              data["pt"],
              data["delp"],
              data["gz"],
              gz_old, 
              pef,
              data["ws"],
              data["q_con"],
              data["cappa"],
              pe,
              data["p_fac"],
              data["scale_m"],
              data["ms"],
              data["dt"],
              data["akap"],
              data["cp"],
              data["ptop"],
              gama,
              origin=(data.ng - 1, data.ng - 1, 0),
              domain=(data.npx + 1, data.npx + 1, data.km) 
        )
    end_time = time.time()
    print("Sum of gz_out = ", np.sum(data["gz_out"]))
    print("Sum of gz     = ", np.sum(data["gz"]))
    print("Sum of pef_out = ", np.sum(data["pef"]))
    print("Sum of pef = ", np.sum(pef))
    vslice = (slice(3,4), slice(3,4), slice(0,5))
    print(pef[vslice], pe[vslice], "pp",pem[vslice], "w2",dm[vslice] )
    print("max diff of gz", np.max(np.abs(data["gz"] - data["gz_out"])))
    pef_slice = (slice(data.ng - 1, data.ng + data.npx - 1), slice(data.ng - 1, data.ng + data.npy - 1), slice(0, data.km))
    print("max relative diff of pef", np.max((np.abs(pef[pef_slice] - data["pef"][pef_slice])) / pef[pef_slice]))
    print('elapsed time (sec) = ', end_time - start_time)
    for i in range(5): #data["gz"].shape[0]):
        for j in range(5): #data["gz"].shape[1]):
            for k in range(3):
                comp = data["gz"][i, j, k]
                ref = data["gz_out"][i, j, k]
                if comp != ref:
                    print(i, j, k, comp, ref, comp - ref)
if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        raise ValueError("Usage: main.py path/to/dataset.nc [backend]")
    file_name = sys.argv[1]
    if len(sys.argv) > 2:
        backend = sys.argv[2]
    else:
        backend = "numpy"
    do_test(file_name, backend)
