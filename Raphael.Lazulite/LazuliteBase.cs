using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public abstract class LazuliteBase<TElement, THost>(MemoryBuffer1D<TElement, Stride1D.Dense> buffer, BufferPool<TElement> pool) : IDisposable
    where TElement : unmanaged 
    where THost : notnull
{
    public MemoryBuffer1D<TElement, Stride1D.Dense> Buffer { get; } = buffer;
    public bool NotDisposable { get; set; } = false;
    public bool Disposed { get; private set; } = false;

    readonly protected BufferPool<TElement> _pool = pool;
    protected Action? _disposeHook;

    public THost ToHost()
    {
        _pool._lctx.Accelerator.Synchronize();
        return Convert(Buffer.GetAsArray1D());
    }

    public void Dispose()
    {
        if (NotDisposable) return;
        _disposeHook?.Invoke();
        _pool.Return();
        Disposed = true;
    }

    public abstract THost Convert(TElement[] raw);
    
}