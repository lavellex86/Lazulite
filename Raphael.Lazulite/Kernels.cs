using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public partial class Compute
{
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>> FillKernel { get; private set; } = new(FillKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>>> ZeroKernel { get; private set; } = new(ZeroKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CopyKernel { get; private set; } = new(CopyKernelImpl);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ConcatKernel { get; private set; } = new(ConcatKernelImpl);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> SliceKernel { get; private set; } =
        new(SliceKernelImpl);
    
    #region Implementations
    private static void FillKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> view, float value) => view[index] = value;
    private static void ZeroKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> view) => view[index] = 0;
    private static void CopyKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> destination, ArrayView1D<float, Stride1D.Dense> source) => destination[index] = source[index];
    private static void ConcatKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
    {
        if (index < a.Length)
            result[index] = a[index];
        else
            result[index] = b[index - a.Length];
    }
    private static void SliceKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> dest, ArrayView1D<float, Stride1D.Dense> source, int start, int end)
    {
        if (index >= start && index < end) dest[index - start] = source[index];
    }
    #endregion
}