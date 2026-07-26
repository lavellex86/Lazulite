using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public abstract class RemoteBase<TElement, THost>(MemoryBuffer1D<TElement, Stride1D.Dense> buffer, BufferPool<TElement> pool) : IDisposable, IRemoteBase<TElement>
    where TElement : unmanaged 
    where THost : notnull
{
    public MemoryBuffer1D<TElement, Stride1D.Dense> Buffer { get; } = buffer;
    public bool NotDisposable { get; set; } = false;
    public bool Disposed { get; private set; } = false;

    public int IntLength { get; } = buffer.IntExtent;

    public BufferPool<TElement> Pool { get; } = pool;
    public LazuliteContext Context => Pool._lctx;
    protected Action? _disposeHook;

    public THost ToHost()
    {
        Context.Synchronize();
        return ConvertToHost(Buffer.GetAsArray1D());
    }

    public virtual RemoteBase<TElement, THost> Set(THost host)
    {
        Buffer.CopyFromCPU(ConvertToRaw(host));
        return this;
    }

    public virtual RemoteBase<TElement, THost> UpdateWith(MemoryBuffer1D<TElement, Stride1D.Dense> source)
    {
        Buffer.CopyFrom(source);
        return this;
    }

    public void Dispose()
    {
        if (NotDisposable) return;
        _disposeHook?.Invoke();
        Pool.Return(Buffer);
        Disposed = true;
    }

    public abstract THost ConvertToHost(TElement[] raw);
    public abstract TElement[] ConvertToRaw(THost host);

    IRemoteBase<TElement> IRemoteBase<TElement>.UpdateWith(MemoryBuffer1D<TElement, Stride1D.Dense> source) => UpdateWith(source);
}