using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

/// <summary>
/// Manages a pool of buffers.
/// </summary>
/// <typeparam name="T"></typeparam>
public class BufferPool<T> : IDisposable where T : unmanaged
{
    private readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<T, Stride1D.Dense>>>> _pool = [];

    public BufferPool()
    {
        foreach (var aidx in Compute.Accelerators.Keys) _pool[aidx] = [];
        Compute.PreHooks.Add(Dispose);
    }
    
    /// <summary>
    /// Clears all buffers in the pool.
    /// </summary>
    public void ClearAll()
    {
        for (int i = 0; i < Compute.Accelerators.Count; i++) ClearAt(i);
    }
    
    /// <summary>
    /// Clears all buffers in the pool for the given accelerator index.
    /// </summary>
    public void ClearAt(int aidx)
    {
        Compute.Synchronize(aidx);
        foreach (var kvp in _pool[aidx])
        {
            while (kvp.Value.TryPop(out var buffer)) buffer.Dispose();
            _pool[aidx].TryRemove(kvp);
        }
        _pool[aidx].Clear();
    }
    
    /// <summary>
    /// Returns a buffer to the pool.
    /// </summary>
    public void Return(MemoryBuffer1D<T, Stride1D.Dense> buffer)
    {
        int aidx = buffer.AcceleratorIndex();
        int size = (int)buffer.Length;

        if (_pool[aidx].TryGetValue(size, out var stack)) stack.Push(buffer);
        else _pool[aidx][size] = new([buffer]);
    }
    /// <summary>
    /// Returns multiple buffers to the pool.
    /// </summary>
    public void Return(params MemoryBuffer1D<T, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    
    /// <summary>
    /// Gets a buffer from the pool.
    /// </summary>
    /// <param name="zero">Whether to clear the buffer of previous data before returning.</param>
    public MemoryBuffer1D<T, Stride1D.Dense> Get(int aidx, int size, bool zero = true) => TryGetFrom(aidx, size, zero);
    /// <summary>
    /// Gets a buffer of the same size and accelerator as the given array view.
    /// </summary>
    /// <param name="zero">Whether to clear the buffer of previous data before returning.</param>
    public MemoryBuffer1D<T, Stride1D.Dense> GetLike(ArrayView1D<T, Stride1D.Dense> view, bool zero = true) => Get(view.AcceleratorIndex(), (int)view.Length, zero);

    /// <summary>
    /// Gets multiple buffers from the pool.
    /// </summary>
    /// <param name="zero">Whether to clear the buffers of previous data before returning.</param>
    public MemoryBuffer1D<T, Stride1D.Dense>[] Get(int aidx, int size, int count, bool zero = true)
    {
        var result = new MemoryBuffer1D<T, Stride1D.Dense>[count];
        for (int i = 0; i < count; i++) result[i] = Get(aidx, size, zero);
        return result;
    }
    
    private static MemoryBuffer1D<T, Stride1D.Dense> Allocate(int aidx, int size) => Compute.Accelerators[aidx].Allocate1D<T>(size);
    private MemoryBuffer1D<T, Stride1D.Dense> TryGetFrom(int aidx, int size, bool zero)
    {
        MemoryBuffer1D<T, Stride1D.Dense> buffer;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
            buffer = stack.TryPop(out var result) ? result : Allocate(aidx, size);
        else buffer = Allocate(aidx, size);

        if (zero) Compute.Call(ZeroKernel, buffer);
        return buffer;
    }

    /// <summary>
    /// Disposes all buffers in the pool.
    /// </summary>
    public void Dispose()
    {
        foreach (var kvp in _pool) 
        foreach (var stack in kvp.Value.Values) 
            while (stack.TryPop(out var buffer)) buffer.Dispose();
    }

    private static readonly KernelStorage<Action<Index1D, ArrayView1D<T, Stride1D.Dense>>> ZeroKernel = new((i, r) => r[i] = default);
}