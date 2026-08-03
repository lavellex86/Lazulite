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