using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraKernels
{
    private static void OuterProductKernelImpl(
        Index1D i,
        TensorArrayView result,
        TensorArrayView x, 
        TensorArrayView y, 
        int m, int n)
    {
        int totalElements = m * n;
        if (i >= totalElements) return;
    
        int row = i / n;
        int col = i % n;
    
        result[i] = x[row] * y[col];
    }
    
    private static void AxpyKernelImpl(Index1D i, 
        TensorArrayView x, 
        TensorArrayView y, 
        float alpha) => y[i] = alpha * x[i] + y[i];
}