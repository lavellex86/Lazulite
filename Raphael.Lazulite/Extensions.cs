using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public static class MemoryBufferExtensions
{
    public static int AcceleratorIndex<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged => Compute.GetAcceleratorIndex(buffer.Accelerator);

    public static MemoryBuffer1D<T, Stride1D.Dense> Set<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, T[] value) where T : unmanaged
    {
        buffer.CopyFromCPU(value);
        return buffer;
    }
    public static MemoryBuffer1D<T, Stride1D.Dense> Copy<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, MemoryBuffer1D<T, Stride1D.Dense> source) where T : unmanaged
    {
        buffer.CopyFrom(source);
        return buffer;
    }
}

public static class ArrayViewExtensions
{
    public static int AcceleratorIndex<T>(this ArrayView1D<T, Stride1D.Dense> view) where T : unmanaged => Compute.GetAcceleratorIndex(view.GetAccelerator());
}

public static class ValueExtensions
{
    public static ScalarValue AsScalar(this Value<,> value) => new(value.Data);
    public static VectorValue AsVector(this Value<,> value) => new(value.Data);
    public static VectorValue AsVector(this Value<,> value) => new(value.Data);
    public static MatrixValue AsMatrix(this Value<,> value) => new(value.Data, value.Shape);
    public static TensorValue3 AsTensorValue3(this Value<,> value) => new(value.Data, value.Shape);

    public static Value<,> NonDisposable<T>(this Value<,> value) where T : notnull
    {
        value.Disposable = false;
        return value;
    }

    public static Value<,> Disposable<T>(this Value<,> value) where T : IDisposable
    {
        value.Disposable = true;
        return value;
    }
    
    public static Value<,> Set<T>(this Value<,> value, Value<,> data) where T : notnull
    {
        value.UpdateWith(data);
        return value;
    }
}

public static class ArrayExtensions
{
    public static float[] ToFloats(this int[] ints) => ints.Select(i => (float)i).ToArray();
}