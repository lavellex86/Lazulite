using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class BufferPool<T>(LazuliteContext lctx) : IDisposable where T : unmanaged
{
    private readonly Dictionary<long, Stack<MemoryBuffer1D<T, Stride1D.Dense>>> _pool = [];
    internal readonly LazuliteContext _lctx = lctx;

    public void Return(MemoryBuffer1D<T, Stride1D.Dense> buffer) => (_pool.TryGetValue(buffer.Length, out var stack) ? stack : _pool[buffer.Length] = []).Push(buffer);
    public void Return(params IEnumerable<MemoryBuffer1D<T, Stride1D.Dense>> buffers)
    {
        foreach (var buffer in buffers) Return(buffer); 
    }
    public MemoryBuffer1D<T, Stride1D.Dense> Get(long length, bool cleared = true)
    {
        var buffer = _pool.GetValueOrDefault(length) is { Count: > 0 } stack ? stack.Pop() : Allocate(length);
        if (cleared) buffer.MemSetToZero();
        return buffer;
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Allocate(long length) => _lctx.Accelerator.Allocate1D<T>(length);

    public void Dispose()
    {
        foreach (var stack in _pool.Values)
        foreach (var buffer in stack)
            buffer.Dispose();
    }
}