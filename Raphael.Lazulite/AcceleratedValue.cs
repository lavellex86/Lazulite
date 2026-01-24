using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public abstract class AcceleratedValue<TData, THost>(MemoryBuffer1D<TData, Stride1D.Dense> data)
    where TData : unmanaged where THost : notnull
{
    public MemoryBuffer1D<TData, Stride1D.Dense> Data { get; } = data;
    public int TotalSize { get; } = (int)data.Length;

    public int AcceleratorIndex { get; } = data.AcceleratorIndex();

    public bool WasDisposed { get; private set; }
    public bool Disposable { get; set; } = true;
    
    public THost ToHost()
    {
        Compute.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }
    public void FromHost(THost value) => Data.CopyFromCPU(Roll(value));
    public void UpdateWith(AcceleratedValue<TData, THost> other) => Data.CopyFrom(other.Data);

    public void Dispose()
    {
        if (WasDisposed || !Disposable) return;
        Pool.Return(Data);
        WasDisposed = true;
    }
    public void Return() => Pool.Return(Data);

    public abstract BufferPool<TData> Pool { get; }
    public abstract THost Unroll(TData[] rolled);
    public abstract TData[] Roll(THost value);
    
    public static implicit operator MemoryBuffer1D<TData, Stride1D.Dense>(AcceleratedValue<TData, THost> acceleratedValue) => acceleratedValue.Data;
}