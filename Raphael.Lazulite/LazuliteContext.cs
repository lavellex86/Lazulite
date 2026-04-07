using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class LazuliteContext : IDisposable
{
    public Accelerator Accelerator { get; }
    internal List<Action> DisposeHooks { get; } = [];

    public void Dispose()
    {
        foreach (var hook in DisposeHooks) hook();
        Accelerator.Dispose();
    }
}