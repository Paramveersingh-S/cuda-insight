import os

if __name__ == "__main__":
    os.system("nvcc kernels/matmul/naive.cu -o naive")
    os.system("cuda-insight profile naive")
