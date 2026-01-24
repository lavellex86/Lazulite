using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

#region ByteArrayValue
public class ByteArrayAcceleratedValue(MemoryBuffer1D<byte, Stride1D.Dense> buffer) : AcceleratedValue<byte, byte[]>(buffer)
{
    public override byte[] Roll(byte[] value) => value;
    public override byte[] Unroll(byte[] rolled) => rolled;

    public override BufferPool<byte> Pool => StaticPool;
    internal static BufferPool<byte> StaticPool => ValueExtensions.BytePool;
}
#endregion
#region IntArrayValue
public class IntArrayAcceleratedValue(MemoryBuffer1D<int, Stride1D.Dense> buffer) : AcceleratedValue<int, int[]>(buffer)
{
    public override int[] Roll(int[] value) => value;
    public override int[] Unroll(int[] rolled) => rolled;
    
    public override BufferPool<int> Pool => StaticPool;
    internal static BufferPool<int> StaticPool => ValueExtensions.IntPool;
}
#endregion
#region LongArrayValue
public class LongArrayAcceleratedValue(MemoryBuffer1D<long, Stride1D.Dense> buffer) : AcceleratedValue<long, long[]>(buffer)
{
    public override long[] Roll(long[] value) => value;
    public override long[] Unroll(long[] rolled) => rolled;
    
    public override BufferPool<long> Pool => StaticPool;
    internal static BufferPool<long> StaticPool => ValueExtensions.LongPool;
}
#endregion
