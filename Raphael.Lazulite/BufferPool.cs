using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class BufferPool<T> : IBufferPool where T : unmanaged
{
    private readonly ConcurrentDictionary<int, ConcurrentDictionary<int, ConcurrentStack<MemoryBuffer1D<T, Stride1D.Dense>>>> _pool = [];
    
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

    private static MemoryBuffer1D<T, Stride1D.Dense> Allocate(int aidx, int size) => Compute.Accelerators[aidx].Allocate1D<T>(size);
    private MemoryBuffer1D<T, Stride1D.Dense> TryGetFrom(int aidx, int size, bool zero)
    {
        MemoryBuffer1D<T, Stride1D.Dense> buffer;
        
        if (_pool[aidx].TryGetValue(size, out var stack))
            buffer = stack.TryPop(out var result) ? result : Allocate(aidx, size);
        else buffer = Allocate(aidx, size);

        if (zero) buffer.MemSetToZero();
        return buffer;
    }

    public void AddAccelerator(int aidx) => _pool[aidx] = [];
}

public interface IBufferPool
{
    public void AddAccelerator(int aidx);
    public void ClearAll();
    public void ClearAt(int aidx);
}