using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public interface IRemoteBase<TElement> : IDisposable
    where TElement : unmanaged
{
    public MemoryBuffer1D<TElement, Stride1D.Dense> Buffer { get; }
    public bool NotDisposable { get; set; }
    public bool Disposed { get; }
    
    public int IntLength { get; }
    public BufferPool<TElement> Pool { get; }
    public LazuliteContext Context { get; }

    public IRemoteBase<TElement> UpdateWith(MemoryBuffer1D<TElement, Stride1D.Dense> source);
}