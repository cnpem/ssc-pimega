import h5py
import numpy

rec  = h5py.File("file_1level.h5", "w")

dset = rec.create_dataset('entry', [100,1,3072,3072], dtype=numpy.int32)

dset[:] = 1

rec.flush()

rec.close()
