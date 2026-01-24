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
    public static BufferPool<float> FloatPool { get; } = new();
    public static BufferPool<double> DoublePool { get; } = new();
    public static BufferPool<int> IntPool { get; } = new();
    public static BufferPool<long> LongPool { get; } = new();
    public static BufferPool<byte> BytePool { get; } = new();

    public static Value<TData, THost> NonDisposable<TData, THost>(this Value<TData, THost> value) where TData : unmanaged where THost : notnull
    {
        value.Disposable = false;
        return value;
    }

    public static Value<TData, THost> Disposable<TData, THost>(this Value<TData, THost> value) where TData : unmanaged where THost : notnull
    {
        value.Disposable = true;
        return value;
    }
    
    public static Value<TData, THost> Set<TData, THost>(this Value<TData, THost> value, Value<TData, THost> data) where TData : unmanaged where THost : notnull
    {
        value.UpdateWith(data);
        return value;
    }
}