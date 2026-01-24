using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.Kernels;

public static class VectorKernels
{
    public static void OuterProductKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> x, 
        ArrayView1D<float, Stride1D.Dense> y, 
        int m, int n) // x is m, y is n, result is m x n
    {
        int totalElements = m * n;
        if (index >= totalElements) return;
    
        int row = index / n;
        int col = index % n;
    
        result[index] = x[row] * y[col];
    }
}