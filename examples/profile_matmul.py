import torch
from cuda_insight.torch_ext import profile_torch_kernel

def main():
    if not torch.cuda.is_available():
        print("CUDA not available for PyTorch. Exiting.")
        return
        
    A = torch.randn(1024, 1024, device='cuda')
    B = torch.randn(1024, 1024, device='cuda')
    
    def my_matmul(a, b):
        # Warmup
        for _ in range(3):
            torch.matmul(a, b)
        return torch.matmul(a, b)
        
    print("Profiling PyTorch matmul...")
    profile_torch_kernel(my_matmul, A, B)

if __name__ == "__main__":
    main()
