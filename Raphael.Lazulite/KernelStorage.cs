using System.Collections.Concurrent;

namespace Raphael.Lazulite;

public class KernelStorage<T>(T action) where T : notnull
{
    public T Action { get; } = action;
    public ConcurrentDictionary<int, T?> Kernels { get; } = [];
    
    public T? this[int index]
    {
        get
        {
            Kernels.TryGetValue(index, out T? item);
            return item;
        }
        set => Kernels[index] = value;
    }
}