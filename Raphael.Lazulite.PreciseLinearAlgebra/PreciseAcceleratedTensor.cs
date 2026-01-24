using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.PreciseLinearAlgebra;

#region PreciseAcceleratedTensor
public abstract class PreciseAcceleratedTensor<T>(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) : AcceleratedValue<double, T>(buffer) where T : notnull
{
    public int[] Shape { get; } = shape;
    public abstract PreciseTensorProxy<T> ToProxy();
    
    public abstract PreciseAcceleratedTensor<T> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape);
    public PreciseAcceleratedTensor<T> Zeros() => Create(Pool.GetLike(this), Shape);
    public PreciseAcceleratedTensor<T> Clone()
    {
        var buffer = Pool.GetLike(this, zero: false);
        buffer.CopyFrom(this);
        return Create(buffer, Shape);
    }
    
    public override BufferPool<double> Pool => Compute.DoublePool;
}
public abstract class PreciseTensorProxy<T>(double[] flatData, int[] shape) where T : notnull
{
    public double[] FlatData { get; } = flatData;
    public int[] Shape { get; } = shape;
    
    protected PreciseTensorProxy(PreciseAcceleratedTensor<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract double Get(int[] index);
    public abstract T ToHost();

    public double this[int i] => FlatData[i];
    
    public static implicit operator double[](PreciseTensorProxy<T> proxy) => proxy.FlatData;
    public static implicit operator T(PreciseTensorProxy<T> proxy) => proxy.ToHost();
}
#endregion
#region PreciseAcceleratedScalar
public class PreciseAcceleratedScalar : PreciseAcceleratedTensor<double>
{
    public PreciseAcceleratedScalar(double value, int aidx) : base(Compute.DoublePool.Get(aidx, 1), []) => FromHost(value);
    public PreciseAcceleratedScalar(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer, []) { }

    public override double Unroll(double[] rolled) => rolled[0];
    public override double[] Roll(double value) => [value];
    
    public override PreciseAcceleratedScalar Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) => new PreciseAcceleratedScalar(buffer);
    public override PreciseScalarProxy ToProxy() => new(this);
}

public class PreciseScalarProxy(PreciseAcceleratedScalar acceleratedValue) : PreciseTensorProxy<double>(acceleratedValue)
{
    public override double Get(int[] index) => FlatData[0];
    public override double ToHost() => FlatData[0];
}
#endregion
#region PreciseAcceleratedMatrix
public class PreciseAcceleratedVector : PreciseAcceleratedTensor<double[]>
{
    public PreciseAcceleratedVector(double[] value, int aidx) : base(Compute.DoublePool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public PreciseAcceleratedVector(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }

    public override double[] Unroll(double[] rolled) => rolled;
    public override double[] Roll(double[] value) => value;
    
    public override PreciseAcceleratedVector Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override PreciseVectorProxy ToProxy() => new(this);
}

public class PreciseVectorProxy(PreciseAcceleratedVector acceleratedValue) : PreciseTensorProxy<double[]>(acceleratedValue)
{
    public override double Get(int[] index) => FlatData[index[0]];
    public override double[] ToHost() => FlatData;
}
#endregion