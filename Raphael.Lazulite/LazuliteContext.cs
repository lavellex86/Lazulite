using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Raphael.Lazulite;

public class LazuliteContext : IDisposable
{
    public Accelerator Accelerator { get; }
    public List<Action> DisposeHooks { get; } = [];

    private Context _ctx;

    public LazuliteContext(bool gpu = true, OptimizationLevel optimization = OptimizationLevel.Release, Accelerator? accelerator = null)
    {
        _ctx = Context.Create(b => b
            .Default()
            .EnableAlgorithms()
            .Optimize(optimization));
        Accelerator = accelerator ?? _ctx.Devices
            .Where(d => gpu ? d is not CPUDevice : d is CPUDevice)
            .OrderBy(RankDevice)
            .First().CreateAccelerator(_ctx);
    }

    public void Dispose()
    {
        foreach (var hook in DisposeHooks) hook();
        Accelerator.Dispose();
    }

    public void Synchronize() => Accelerator.Synchronize();
    public string AcceleratorName => Accelerator.Name;

    private int RankDevice(Device d)
    {
        return d.AcceleratorType switch
        {
            AcceleratorType.Cuda => 0,
            AcceleratorType.OpenCL => 1,
            AcceleratorType.CPU => 2,
            _ => 3
        };
    }
}