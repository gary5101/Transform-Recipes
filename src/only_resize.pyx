cimport cython
cimport numpy as np
import numpy as np

np.import_array()


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def resize(_inx,weightsx,indicesx,o_height,o_width, dim, outx):

    cdef double[:, :] _in = np.asarray(_inx, dtype=np.double)
    cdef double[:, :] weights = np.asarray(weightsx, dtype=np.double)
    cdef double[:] out = np.asarray(outx, dtype=np.double)
    cdef long[:,:] indices = np.asarray(indicesx, dtype=np.long)

    cdef index  = 0;
    cdef val = 0;
    cdef w_width = weights.shape[1]

    # print(weights.shape,_in.shape,indices.shape,out.shape)
    cdef np.npy_intp i, j, s_id
    if dim == 0:
        for i in range(o_height):
            for j in range(o_width):
                index = i*o_width + j;
                val   = 0;
                for s_id in range(w_width):
                    val += _in[ indices[i,s_id], j]*weights[i,s_id];
                out[index] = val;
    else:
        for i in range(o_height):
            for j in range(o_width):
                index = i*o_width + j;
                val   = 0;
                for s_id in range(w_width):
                    val += _in[i,indices[j,s_id]]*weights[j,s_id];
                out[index] = val;
    return out

