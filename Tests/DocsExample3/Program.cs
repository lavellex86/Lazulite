using ILGPU;
using ILGPU.Runtime;
using Raphael.Lazulite;
using System.Drawing;

using var lctx = new LazuliteContext();
using var pool = new BufferPool<byte>(lctx);

using var red = new RemoteColor(pool.Get(4), pool)
    .Set(Color.Red);
using var blue = new RemoteColor(pool.Get(4), pool)
    .Set(Color.Blue);

using var redCopy = new RemoteColor(pool.Get(4), pool)
    .Set(red.Buffer);

Console.WriteLine($"{redCopy.Length}, {redCopy.Disposed}, {redCopy.Disposable}");

public class RemoteColor(MemoryBuffer1D<byte, Stride1D.Dense> buffer, BufferPool<byte> pool) : RemoteBase<byte, Color>(buffer, pool)
{
    protected override Color ConvertToHost(byte[] raw) => Color.FromArgb(raw[0], raw[1], raw[2], raw[3]);
    protected override byte[] ConvertToRaw(Color host) => [host.A, host.R, host.G, host.B];
}