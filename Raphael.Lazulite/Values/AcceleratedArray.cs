using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

#region AcceleratedByteArray
/// <summary>
/// Represents an array of <c>byte</c>s on an accelerator.
/// </summary>
public class AcceleratedByteArray(MemoryBuffer1D<byte, Stride1D.Dense> buffer) : AcceleratedValue<byte, byte[]>(buffer)
{
    public AcceleratedByteArray(byte[] array, int aidx = -1) : this(Compute.BytePool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedByteArray(int length, int aidx = -1) : this(Compute.BytePool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override byte[] Transform(byte[] flattened) => flattened;
    public override BufferPool<byte> Pool => Compute.BytePool;
}
#endregion
#region AcceleratedIntArray
/// <summary>
/// Represents an array of <c>int</c>s on an accelerator.
/// </summary>
public class AcceleratedIntArray(MemoryBuffer1D<int, Stride1D.Dense> buffer) : AcceleratedValue<int, int[]>(buffer)
{
    public AcceleratedIntArray(int[] array, int aidx = -1) : this(Compute.IntPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedIntArray(int length, int aidx = -1) : this(Compute.IntPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override int[] Transform(int[] flattened) => flattened;
    public override BufferPool<int> Pool => Compute.IntPool;
}
#endregion
#region AcceleratedUnsignedIntArray
/// <summary>
/// Represents an array of <c>uint</c>s on an accelerator.
/// </summary>
public class AcceleratedUnsignedIntArray(MemoryBuffer1D<uint, Stride1D.Dense> buffer) : AcceleratedValue<uint, uint[]>(buffer)
{
    public AcceleratedUnsignedIntArray(uint[] array, int aidx = -1) : this(Compute.UnsignedIntPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedUnsignedIntArray(int length, int aidx = -1) : this(Compute.UnsignedIntPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override uint[] Transform(uint[] flattened) => flattened;
    public override BufferPool<uint> Pool => Compute.UnsignedIntPool;
}
#endregion
#region AcceleratedLongArray
/// <summary>
/// Represents an array of <c>long</c>s on an accelerator.
/// </summary>
public class AcceleratedLongArray(MemoryBuffer1D<long, Stride1D.Dense> buffer) : AcceleratedValue<long, long[]>(buffer)
{
    public AcceleratedLongArray(long[] array, int aidx = -1) : this(Compute.LongPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedLongArray(int length, int aidx = -1) : this(Compute.LongPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override long[] Transform(long[] flattened) => flattened;
    public override BufferPool<long> Pool => Compute.LongPool;
}
#endregion
#region AcceleratedUnsignedLongArray
/// <summary>
/// Represents an array of <c>ulong</c>s on an accelerator.
/// </summary>
public class AcceleratedUnsignedLongArray(MemoryBuffer1D<ulong, Stride1D.Dense> buffer) : AcceleratedValue<ulong, ulong[]>(buffer)
{
    public AcceleratedUnsignedLongArray(ulong[] array, int aidx = -1) : this(Compute.UnsignedLongPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedUnsignedLongArray(int length, int aidx = -1) : this(Compute.UnsignedLongPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override ulong[] Transform(ulong[] flattened) => flattened;
    public override BufferPool<ulong> Pool => Compute.UnsignedLongPool;
}
#endregion
#region AcceleratedFloatArray
/// <summary>
/// Represents an array of <c>float</c>s on an accelerator.
/// </summary>
public class AcceleratedFloatArray(MemoryBuffer1D<float, Stride1D.Dense> buffer) : AcceleratedValue<float, float[]>(buffer)
{
    public AcceleratedFloatArray(float[] array, int aidx = -1) : this(Compute.FloatPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedFloatArray(int length, int aidx = -1) : this(Compute.FloatPool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override float[] Transform(float[] flattened) => flattened;
    public override BufferPool<float> Pool => Compute.FloatPool;
}
#endregion
#region AcceleratedDoubleArray
/// <summary>
/// Represents an array of <c>double</c>s on an accelerator.
/// </summary>
public class AcceleratedDoubleArray(MemoryBuffer1D<double, Stride1D.Dense> buffer) : AcceleratedValue<double, double[]>(buffer)
{
    public AcceleratedDoubleArray(double[] array, int aidx = -1) : this(Compute.DoublePool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, array.Length).Set(array)) { }
    public AcceleratedDoubleArray(int length, int aidx = -1) : this(Compute.DoublePool.Get(aidx == -1 ? Compute.RequestOptimalAccelerator() : aidx, length)) { }
    public override double[] Transform(double[] flattened) => flattened;
    public override BufferPool<double> Pool => Compute.DoublePool;
}
#endregion