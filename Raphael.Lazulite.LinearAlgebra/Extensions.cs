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
    
    public static AcceleratedScalar AsScalar(this AcceleratedTensor<float> tensor)
    {
        if (tensor is AcceleratedScalar scalar) return scalar;
        return new AcceleratedScalar(tensor);
    }

    public static AcceleratedVector AsVector(this AcceleratedTensor<float[]> tensor)
    {
        if (tensor is AcceleratedVector vector) return vector;
        return new AcceleratedVector(tensor);
    }

    public static AcceleratedMatrix AsMatrix(this AcceleratedTensor<float[,]> tensor)
    {
        if (tensor is AcceleratedMatrix matrix) return matrix;
        return new AcceleratedMatrix(tensor, tensor.Shape);
    }

    public static AcceleratedTensor<T> AsTensor<T>(this AcceleratedValue<float, T> value) where T : notnull
    {
        if (value is AcceleratedTensor<T> tensor) return tensor;
        throw new InvalidOperationException("Value is not a tensor.");
    }
}

public static class MemoryBufferExtensions
{
    public static void Return(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.FloatPool.Return(buffer);

    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(this MemoryBuffer1D<float, Stride1D.Dense> alike, Action<MemoryBuffer1D<float, Stride1D.Dense>> action)
    {
        var result = Compute.FloatPool.GetLike(alike);
        action(result);
        return result;
    }
}

public static class ArrayViewExtensions
{
    public static MemoryBuffer1D<float, Stride1D.Dense> Encase(this TensorArrayView alike, Action<TensorArrayView> action)
    {
        var result = Compute.FloatPool.Get(alike.AcceleratorIndex(), (int)alike.Length);
        action(result);
        return result;
    }
}