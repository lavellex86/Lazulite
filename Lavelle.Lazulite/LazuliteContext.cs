using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;

namespace Lavelle.Lazulite;

/// <summary>
/// Holds all information regarding the compute device and manages extension libraries.
/// </summary>
public class LazuliteContext : IDisposable
{
    /// <summary>
    /// The ILGPU accelerator underlying Lazulite.
    /// </summary>
    public Accelerator Accelerator { get; }
    /// <summary>
    /// A set of actions to run upon disposal.
    /// </summary>
    public List<Action> DisposeHooks { get; } = [];

    private Context _ctx;

    /// <summary>
    /// Creates a <see cref="LazuliteContext"/>.
    /// </summary>
    /// <param name="gpu">Whether to look for a GPU accelerator or a CPU accelerator.</param>
    /// <param name="optimization">The level of optimization ILGPU should use to compile kernels. Release mode by default.</param>
    /// <param name="accelerator">The ILGPU accelerator to use.</param>
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

    /// <summary>
    /// Calls all <see cref="DisposeHooks"/> and disposes of the ILGPU accelerator.
    /// </summary>
    public void Dispose()
    {
        foreach (var hook in DisposeHooks) hook();
        Accelerator.Dispose();
    }

    /// <summary>
    /// Synchronizes the runtime with the compute device.
    /// </summary>
    public void Synchronize() => Accelerator.Synchronize();
    /// <summary>
    /// The name of the compute device.
    /// </summary>
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