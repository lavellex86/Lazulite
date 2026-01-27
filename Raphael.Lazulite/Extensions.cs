using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public static class MemoryBufferExtensions
{
    /// <summary>
    /// Returns the accelerator index of a given buffer.
    /// </summary>
    public static int AcceleratorIndex<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer) where T : unmanaged => Compute.GetAcceleratorIndex(buffer.Accelerator);

    /// <summary>
    /// Sets the contents of a buffer to the given value.
    /// </summary>
    public static MemoryBuffer1D<T, Stride1D.Dense> Set<T>(this MemoryBuffer1D<T, Stride1D.Dense> buffer, T[] value) where T : unmanaged
    {
        buffer.CopyFromCPU(value);
        return buffer;
    }
    /// <summary>
    /// Copies the contents of a buffer to another buffer.
    /// </summary>
    public static MemoryBuffer1D<T, Stride1D.Dense> Copy<T>(this MemoryBuffer1D<T, Stride1D.Dense> dest, MemoryBuffer1D<T, Stride1D.Dense> source) where T : unmanaged
    {
        dest.CopyFrom(source);
        return dest;
    }
    
    public static void Return(this MemoryBuffer1D<float, Stride1D.Dense> buffer) => Compute.FloatPool.Return(buffer);
    public static void Return(this MemoryBuffer1D<double, Stride1D.Dense> buffer) => Compute.DoublePool.Return(buffer);
    public static void Return(this MemoryBuffer1D<int, Stride1D.Dense> buffer) => Compute.IntPool.Return(buffer);
    public static void Return(this MemoryBuffer1D<uint, Stride1D.Dense> buffer) => Compute.UnsignedIntPool.Return(buffer);
    public static void Return(this MemoryBuffer1D<long, Stride1D.Dense> buffer) => Compute.LongPool.Return(buffer);
    public static void Return(this MemoryBuffer1D<ulong, Stride1D.Dense> buffer) => Compute.UnsignedLongPool.Return(buffer);
    public static void Return(this MemoryBuffer1D<byte, Stride1D.Dense> buffer) => Compute.BytePool.Return(buffer);
}

public static class ArrayViewExtensions
{
    /// <summary>
    /// Returns the accelerator index of a given array view.
    /// </summary>
    public static int AcceleratorIndex<T>(this ArrayView1D<T, Stride1D.Dense> view) where T : unmanaged => Compute.GetAcceleratorIndex(view.GetAccelerator());
}

public static class ValueExtensions
{
    /// <summary>
    /// Marks the given value as non-disposable.
    /// </summary>
    public static AcceleratedValue<TData, THost> NonDisposable<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.Disposable = false;
        return acceleratedValue;
    }

    /// <summary>
    /// Marks the given value as disposable.
    /// </summary>
    public static AcceleratedValue<TData, THost> Disposable<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.Disposable = true;
        return acceleratedValue;
    }
    
    /// <summary>
    /// Updates the contents of the given value and returns the updated value.
    /// </summary>
    public static AcceleratedValue<TData, THost> Set<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue, AcceleratedValue<TData, THost> update) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.UpdateWith(update);
        return acceleratedValue;
    }
}