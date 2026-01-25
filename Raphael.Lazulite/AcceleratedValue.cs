using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

/// <summary>
/// Wraps an underlying data buffer.
/// </summary>
public abstract class AcceleratedValue<TData, THost>(MemoryBuffer1D<TData, Stride1D.Dense> data) : IDisposable
    where TData : unmanaged where THost : notnull
{
    /// <summary>
    /// The underlying data buffer.
    /// </summary>
    public MemoryBuffer1D<TData, Stride1D.Dense> Data { get; } = data;
    
    /// <summary>
    /// The total size of the underlying data buffer.
    /// </summary>
    public int TotalSize { get; } = (int)data.Length;

    /// <summary>
    /// The index of the accelerator that owns the underlying data buffer.
    /// </summary>
    public int AcceleratorIndex { get; } = data.AcceleratorIndex();

    /// <summary>
    /// Whether the underlying data buffer has been disposed of.
    /// </summary>
    public bool WasDisposed { get; private set; }
    /// <summary>
    /// Whether this value has been marked as non-disposable.
    /// </summary>
    public bool Disposable { get; set; } = true;
    
    /// <summary>
    /// Returns the <typeparamref name="THost"/> representation of the underlying data buffer.
    /// </summary>
    /// <remarks>This is expensive and will synchronize! Use sparingly.</remarks>
    public THost ToHost()
    {
        Compute.Synchronize(AcceleratorIndex);
        return Unroll(Data.View.GetAsArray1D());
    }
    /// <summary>
    /// Updates the contents of this value with the contents of the given value.
    /// </summary>
    public void UpdateWith(AcceleratedValue<TData, THost> other) => Data.CopyFrom(other.Data);

    /// <summary>
    /// Returns the underlying data buffer to the pool if this value was marked as disposable.
    /// </summary>
    public void Dispose()
    {
        if (WasDisposed || !Disposable) return;
        Pool.Return(Data);
        WasDisposed = true;
    }

    /// <summary>
    /// The pool that owns the underlying data buffer.
    /// </summary>
    public abstract BufferPool<TData> Pool { get; }
    
    /// <summary>
    /// Transforms a <typeparamref name="TData"/> array into a <typeparamref name="THost"/>./>
    /// </summary>
    public abstract THost Unroll(TData[] rolled);
    
    public static implicit operator MemoryBuffer1D<TData, Stride1D.Dense>(AcceleratedValue<TData, THost> acceleratedValue) => acceleratedValue.Data;
    public static implicit operator ArrayView1D<TData, Stride1D.Dense>(AcceleratedValue<TData, THost> acceleratedValue) => acceleratedValue.Data.View;
}