# Lazulite
A fast & powerful scientific computing stack for C#, built on GPU acceleration.
## `Lavelle.Lazulite` Overview
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
## `Lavelle.Linalg32`, and `Lavelle.Calc32` Overview
`Linalg32` and `Calc32` host various extension methods and classes implementing Lazulite's primitives, including integration methods and tensor operations.

You can find the docs for all three packages [here](https://lavelle.gitbook.io/lazulite-documentation).
