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
    public static AcceleratedValue<TData, THost> NonDisposable<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.Disposable = false;
        return acceleratedValue;
    }

    public static AcceleratedValue<TData, THost> Disposable<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.Disposable = true;
        return acceleratedValue;
    }
    
    public static AcceleratedValue<TData, THost> Set<TData, THost>(this AcceleratedValue<TData, THost> acceleratedValue, AcceleratedValue<TData, THost> data) where TData : unmanaged where THost : notnull
    {
        acceleratedValue.UpdateWith(data);
        return acceleratedValue;
    }
}