using ILGPU;
using ILGPU.Runtime;

namespace Lavelle.Lazulite;

/// <summary>
/// A pool of reusable buffers.
/// </summary>
/// <typeparam name="T">The buffer type to keep</typeparam>
/// <param name="lctx">The Lazulite context over the pool.</param>
public class BufferPool<T>(LazuliteContext lctx) : IDisposable where T : unmanaged
{
    private readonly Dictionary<long, Stack<MemoryBuffer1D<T, Stride1D.Dense>>> _pool = [];
    internal readonly LazuliteContext _lctx = lctx;

    /// <summary>
    /// Returns a buffer to the pool.
    /// </summary>
    public void Return(MemoryBuffer1D<T, Stride1D.Dense> buffer) => (_pool.TryGetValue(buffer.Length, out var stack) ? stack : _pool[buffer.Length] = []).Push(buffer);
    /// <summary>
    /// Returns a set of buffers to the pool.
    /// </summary>
    public void Return(params IEnumerable<MemoryBuffer1D<T, Stride1D.Dense>> buffers)
    {
        foreach (var buffer in buffers) Return(buffer); 
    }
    /// <summary>
    /// Retrieves a buffer of length <paramref name="length"/> from the pool.
    /// </summary>
    /// <param name="length">The length of the buffer.</param>
    /// <param name="cleared">Whether to clear (zero) the buffer out before returning.</param>
    /// <returns></returns>
    public MemoryBuffer1D<T, Stride1D.Dense> Get(long length, bool cleared = false)
    {
        var buffer = _pool.GetValueOrDefault(length) is { Count: > 0 } stack ? stack.Pop() : Allocate(length);
        if (cleared) buffer.MemSetToZero();
        return buffer;
    }

    private MemoryBuffer1D<T, Stride1D.Dense> Allocate(long length) => _lctx.Accelerator.Allocate1D<T>(length);

    /// <summary>
    /// Disposes of all the buffers in the pool.
    /// </summary>
    public void Dispose()
    {
        foreach (var stack in _pool.Values)
        foreach (var buffer in stack)
            buffer.Dispose();
    }
}