import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

def matrix_multiply_gpu(A, B):
    N = A.shape[0]

    # Allocate memory on the GPU
    A_gpu = drv.mem_alloc(A.nbytes)
    B_gpu = drv.mem_alloc(B.nbytes)
    C_gpu = drv.mem_alloc(C.nbytes)

    # Copy matrices to the GPU
    drv.memcpy_htod(A_gpu, A)
    drv.memcpy_htod(B_gpu, B)

    # Set block and grid dimensions
    block_dim = (16, 16, 1)
    grid_dim = (int(N / block_dim[0]), int(N / block_dim[1]), 1)

    # Launch the kernel
    matrixMulKernel[grid_dim, block_dim](A_gpu, B_gpu, C_gpu, N)

    # Copy the result back to the CPU
    drv.memcpy_dtoh(C, C_gpu)

    return C

# Example usage:
N = 1024
A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

C_gpu = matrix_multiply_gpu(A, B)

# Print the result or use it for further computations
print(C_gpu)
