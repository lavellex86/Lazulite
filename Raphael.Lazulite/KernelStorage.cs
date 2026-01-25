using System.Collections.Concurrent;

namespace Raphael.Lazulite;

/// <summary>
/// Stores kernel instances by their accelerator index.
/// </summary>
public class KernelStorage<T>(T action) where T : notnull
{
    internal T Action { get; } = action;
    internal ConcurrentDictionary<int, T?> Kernels { get; } = [];
    
    internal T? this[int index]
    {
        get
        {
            Kernels.TryGetValue(index, out T? item);
            return item;
        }
        set => Kernels[index] = value;
    }
}