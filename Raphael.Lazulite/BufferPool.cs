using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class BufferPool<T> : IBufferPool, IDisposable where T : unmanaged
{
    private readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<T, Stride1D.Dense>>>> _pool = [];

    public BufferPool()
    {
        foreach (var aidx in Compute.Accelerators.Keys) _pool[aidx] = [];
        Compute.BufferPoolHooks.Add(this);
    }
    
    public void ClearAll()
    {
        for (int i = 0; i < Compute.Accelerators.Count; i++) ClearAt(i);
    }
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
    
    public void Return(MemoryBuffer1D<T, Stride1D.Dense> buffer)
    {
        int aidx = buffer.AcceleratorIndex();
        int size = (int)buffer.Length;

        if (_pool[aidx].TryGetValue(size, out var stack)) stack.Push(buffer);
        else _pool[aidx][size] = new([buffer]);
    }
    public void Return(params MemoryBuffer1D<T, Stride1D.Dense>[] buffers)
    {
        foreach (var buffer in buffers) Return(buffer);
    } 
    
    public MemoryBuffer1D<T, Stride1D.Dense> Get(int aidx, int size, bool zero = true) => TryGetFrom(aidx, size, zero);
    public MemoryBuffer1D<T, Stride1D.Dense> GetLike(MemoryBuffer1D<T, Stride1D.Dense> buffer, bool zero = true) => Get(buffer.AcceleratorIndex(), (int)buffer.Length, zero);

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

    public void Dispose()
    {
        foreach (var kvp in _pool) 
        foreach (var stack in kvp.Value.Values) 
            while (stack.TryPop(out var buffer)) buffer.Dispose();
    }

    private readonly KernelStorage<Action<Index1D, ArrayView1D<T, Stride1D.Dense>>> ZeroKernel = new((i, r) => r[i] = default);
}

public interface IBufferPool
{
    public void ClearAll();
    public void ClearAt(int aidx);
}