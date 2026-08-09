# Lazulite
A fast & powerful scientific computing stack for C#, built on GPU acceleration. 
## `Lavelle.Lazulite`
Lazulite is a GPU acceleration library built on [ILGPU](ilgpu.net), handling automatic buffer pooling and lazy kernel initialization. 
It uses remotes (objects stored on the compute device) to handle computation entirely on the compute device before syncing back, minimizing the sync costs.
With it, GPU programming becomes simple:
```csharp
using ILGPU;
using ILGPU.Runtime;
using Lavelle.Lazulite;

using var lctx = new LazuliteContext(); 
using var pool = new BufferPool<int>(lctx);

using var remote = new RemoteIntArray(pool.Get(3), pool) 
    .Set([1, 2, 3]);

var kernel = new LazuliteKernel<Action<Index1D, ArrayView1D<int, Stride1D.Dense>>>((i, arr) => arr[i] += 1, lctx);
kernel.Call(3, remote.Buffer);

var returned = remote.Get();
Console.WriteLine($"[{returned[0]}, {returned[1]}, {returned[2]}]");

class RemoteIntArray(MemoryBuffer1D<int, Stride1D.Dense> buffer, BufferPool<int> pool) 
    : RemoteBase<int, int[]>(buffer, pool) 
{
    protected override int[] ConvertToHost(int[] raw) => raw;
    protected override int[] ConvertToRaw(int[] host) => host;
}
```
Remotes are managed by Lazulite itself; all you need to implement is the conversion from the underlying flattened array of elements (for exmample, a `byte[]`) to the object you're representing and vice versa.
Lazulite serves as the base of the Lazulite stack, providing easy GPU acceleration to strengthen the rest of the libraries.
You can find the docs for the whole Lazulite stack [here](https://lavelle.gitbook.io/lazulite-documentation).
## `Lavelle.Linalg32`
`Linalg32` is a float-based library handling tensor operations, implementing elementwise ops as well as methods like the determinant and decompositions.
Using it is easy:
```
using Lavelle.Lazulite;
using Lavelle.Linalg32;

using var lctx = new LazuliteContext().EnableLinalg32();

using var a = lctx.GetVector(3);
using var b = lctx.GetVector(3);
using var sum = lctx.Add(a, b);
using var product = lctx.Multiply(a, b);
```
`Linalg32` also integrates cuBLAS for standard operations, dramatically increasing compute speed.
## `Lavelle.Calc32`
`Calc32` implements float-based calculus methods for tensors, allowing numerical differentation and integration on the GPU:
```csharp
using Lavelle.Calc32;
using Lavelle.Lazulite;
using Lavelle.Linalg32;

var lctx = new LazuliteContext();
var cctx = new CalcContext(lctx);

var f1 = new RemoteVector[]
{
    lctx.GetVector(1).Set([1]).AsVector(),
    lctx.GetVector(1).Set([2]).AsVector(),
    lctx.GetVector(1).Set([3]).AsVector(),
    lctx.GetVector(1).Set([4]).AsVector()
};
var df1 = cctx.Differentiate(f1, 0.01f);

var f2 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 1, 1]).AsVector(),
    lctx.GetVector(3).Set([2, 2, 2]).AsVector(),
    lctx.GetVector(3).Set([1, 1, 1]).AsVector()
};
var initialF2 = lctx.GetVector(3).Set([0, 0, 0]).AsVector();
var F2 = cctx.EulerIntegrate(f2, initialF2, 0.01f);
```
