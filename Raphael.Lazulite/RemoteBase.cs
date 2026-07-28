using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

/// <summary>
/// Represents an object on the compute device.
/// </summary>
/// <typeparam name="TElement">The unmanaged data type stored on the compute device.</typeparam>
/// <typeparam name="THost">The class represented by the remote.</typeparam>
/// <param name="buffer">The memory buffer of <typeparamref name="TElement"/>s that hold the remote object.</param>
/// <param name="pool">The buffer pool this object belongs too.</param>
public abstract class RemoteBase<TElement, THost>(MemoryBuffer1D<TElement, Stride1D.Dense> buffer, BufferPool<TElement> pool) : IDisposable
    where TElement : unmanaged 
    where THost : notnull
{
    /// <summary>
    /// The memory buffer of <typeparamref name="TElement"/>s that hold the remote object
    /// </summary>
    public MemoryBuffer1D<TElement, Stride1D.Dense> Buffer { get; } = buffer;
    /// <summary>
    /// Whether this object is disposable. When true, it will be disposed of or returned to the pool.
    /// </summary>
    public bool Disposable { get; set; } = true;
    /// <summary>
    /// Whether this object has been disposed of.
    /// </summary>
    public bool Disposed { get; private set; } = false;

    /// <summary>
    /// The number of <typeparamref name="TElement"/>s in the underlying memory buffer.
    /// </summary>
    public int Length { get; } = buffer.IntExtent;

    /// <summary>
    /// The buffer pool this object belongs too.
    /// </summary>
    public BufferPool<TElement> Pool { get; } = pool;
    /// <summary>
    /// The Lazulite context this object is under.
    /// </summary>
    public LazuliteContext Context => Pool._lctx;
    /// <summary>
    /// An action to call on disposal.
    /// </summary>
    protected Action? _disposeHook;

    /// <summary>
    /// Gets the remote object.
    /// </summary>
    public THost Get()
    {
        Context.Synchronize();
        return ConvertToHost(Buffer.GetAsArray1D());
    }

    /// <summary>
    /// Sets the remote object.
    /// </summary>
    /// <param name="host">The object to set the remote object to.</param>
    public virtual RemoteBase<TElement, THost> Set(THost host)
    {
        Buffer.CopyFromCPU(ConvertToRaw(host));
        return this;
    }

    /// <summary>
    /// Sets the remote object using a memory buffer.
    /// </summary>
    /// <param name="source">The memory buffer to copy from.</param>
    public virtual RemoteBase<TElement, THost> Set(MemoryBuffer1D<TElement, Stride1D.Dense> source)
    {
        Buffer.CopyFrom(source);
        return this;
    }

    /// <summary>
    /// Returns the buffer to the pool.
    /// </summary>
    public void Dispose()
    {
        if (!Disposable) return;
        _disposeHook?.Invoke();
        Pool.Return(Buffer);
        Disposed = true;
    }

    /// <summary>
    /// Converts a flattened buffer to a target object.
    /// </summary>
    /// <param name="raw">The flattened buffer to convert from.</param>
    protected abstract THost ConvertToHost(TElement[] raw);
    /// <summary>
    /// Converts an object into a flattened buffer.
    /// </summary>
    /// <param name="host">The object to convert from.</param>
    protected abstract TElement[] ConvertToRaw(THost host);
}