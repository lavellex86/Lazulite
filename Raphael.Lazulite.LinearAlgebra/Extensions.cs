using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static class AcceleratedTensorExtensions
{
    public static AcceleratedTensor<T> Encase<T>(this AcceleratedTensor<T> alike, Action<AcceleratedTensor<T>> action) where T : notnull
    {
        var result = alike.CreateAlike(alike.Pool.GetLike(alike));
        action(result);
        return result;
    }
}

public static class MemoryBufferExtensions
{
    public static BufferPool<float> Pool(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => ValueExtensions.FloatPool;
    public static void Return(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => buffer.Pool().Return(buffer);

    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(this MemoryBuffer1D<float, Stride1D.Dense> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> action)
    {
        var result = ValueExtensions.FloatPool.GetLike(alike);
        action(result);
        return result;
    }
}