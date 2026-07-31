using ILGPU;
using Raphael.Lazulite;
using Raphael.Linalg32;

using var lctx1 = new LazuliteContext(gpu: true, optimization: OptimizationLevel.Debug);
using var lctx2 = new LazuliteContext(false, OptimizationLevel.O0);
using var lctx3 = new LazuliteContext(true, OptimizationLevel.O1);
using var lctx4 = new LazuliteContext(false, OptimizationLevel.O2);
using var lctx5 = new LazuliteContext(true, OptimizationLevel.Release);

using var lctx = new LazuliteContext()
    .EnableLinalg32();

lctx.DisposeHooks.Add(() => Console.WriteLine("Disposing!"));

lctx.Synchronize();

Console.WriteLine(lctx.AcceleratorName);
Console.WriteLine(lctx.Accelerator.AcceleratorType);