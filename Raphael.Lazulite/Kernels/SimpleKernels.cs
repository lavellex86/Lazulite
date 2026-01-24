using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.Kernels;

public static class SimpleKernels
{
    public static void FillKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view, float value) => view[index] = value;
    public static void ZeroKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> view) => view[index] = 0;
    
    public static void CopyKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> destination, ArrayView1D<float, Stride1D.Dense> source) => destination[index] = source[index];
    public static void ConcatKernel(Index1D index, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
    {
        if (index < a.Length)
            result[index] = a[index];
        else
            result[index] = b[index - a.Length];
    }

    public static void SliceKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> dest,
        ArrayView1D<float, Stride1D.Dense> source,
        int start, int end)
    {
        if (index >= start && index < end) dest[index - start] = source[index];
    }
}