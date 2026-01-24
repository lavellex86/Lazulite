using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraSuite
{
    public static void OuterProductKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> x, 
        ArrayView1D<float, Stride1D.Dense> y, 
        int m, int n)
    {
        int totalElements = m * n;
        if (index >= totalElements) return;
    
        int row = index / n;
        int col = index % n;
    
        result[index] = x[row] * y[col];
    }
    
    public static void AxpyKernelImpl(Index1D index, 
        ArrayView1D<float, Stride1D.Dense> x, 
        ArrayView1D<float, Stride1D.Dense> y, 
        float alpha) => y[index] = alpha * x[index] + y[index];
}