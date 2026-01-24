using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

#region AcceleratedTensor
public abstract class AcceleratedTensor<T>(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : AcceleratedValue<float, T>(buffer) where T : notnull
{
    public int[] Shape { get; } = shape;
    public abstract TensorProxy<T> ToProxy();
    
    public abstract AcceleratedTensor<T> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape);
    public AcceleratedTensor<T> Zeros() => Create(Pool.GetLike(this), Shape);
    public AcceleratedTensor<T> Clone()
    {
        var buffer = Pool.GetLike(this, zero: false);
        buffer.CopyFrom(this);
        return Create(buffer, Shape);
    }
    public AcceleratedTensor<T> CreateAlike(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Create(buffer, Shape);

    public override BufferPool<float> Pool => StaticPool;
    internal static BufferPool<float> StaticPool => ValueExtensions.FloatPool;
}

public abstract class TensorProxy<T>(float[] flatData, int[] shape) where T : notnull
{
    public float[] FlatData { get; } = flatData;
    public int[] Shape { get; } = shape;
    
    protected TensorProxy(AcceleratedTensor<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract float Get(int[] index);
    public abstract T ToHost();

    public float this[int i] => FlatData[i];
    
    public static implicit operator float[](TensorProxy<T> proxy) => proxy.FlatData;
    public static implicit operator T(TensorProxy<T> proxy) => proxy.ToHost();
}
#endregion

#region AcceleratedScalar
public class AcceleratedScalar : AcceleratedTensor<float>
{
    public AcceleratedScalar(float value, int aidx) : base(StaticPool.Get(aidx, 1), []) => FromHost(value);
    public AcceleratedScalar(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, []) { }
    
    public override float Unroll(float[] rolled) => rolled[0];
    public override float[] Roll(float value) => [value];
    public override AcceleratedScalar Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override ScalarProxy ToProxy() => new(this);
}

public class ScalarProxy(AcceleratedScalar acceleratedValue) : TensorProxy<float>(acceleratedValue)
{
    public override float Get(int[] index) => FlatData[0];
    public override float ToHost() => FlatData[0];
}
#endregion
#region AcceleratedVector
public class AcceleratedVector : AcceleratedTensor<float[]>
{
    public AcceleratedVector(float[] value, int aidx) : base(StaticPool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public AcceleratedVector(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override AcceleratedVector Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override VectorProxy ToProxy() => new(this);
}

public class VectorProxy(AcceleratedVector acceleratedValue) : TensorProxy<float[]>(acceleratedValue)
{
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}
#endregion
#region AcceleratedMatrix
public class AcceleratedMatrix : AcceleratedTensor<float[,]>
{
    public AcceleratedMatrix(float[,] value, int aidx) : base(
        StaticPool.Get(aidx, value.GetLength(0) * value.GetLength(1)),
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public AcceleratedMatrix(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }
    
    public override float[] Roll(float[,] value) => MatrixProxy.Roll(value);
    public override float[,] Unroll(float[] rolled) => MatrixProxy.Unroll(rolled, Shape[1]);

    public override AcceleratedMatrix Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override MatrixProxy ToProxy() => new(this);
}

public class MatrixProxy(AcceleratedMatrix acceleratedValue) : TensorProxy<float[,]>(acceleratedValue)
{
    public float this[int i, int j] => FlatData[KernelProgramming.MatrixIndexOf(i, j, Shape[1])];
    public override float Get(int[] index) => this[index[0], index[1]];
    public override float[,] ToHost() => Unroll(FlatData, Shape[1]);
    
    public static float[] Roll(float[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        var vector = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) vector[KernelProgramming.MatrixIndexOf(i, j, cols)] = value[i, j];
        return vector;
    }
    
    public static float[,] Unroll(float[] rolled, int cols)
    {
        var rows = rolled.Length / cols;
        var matrix = new float[rows, cols];
        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            matrix[i, j] = rolled[KernelProgramming.MatrixIndexOf(i, j, cols)];
        return matrix;
    }
}
#endregion

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
    
    public override BufferPool<double> Pool => StaticPool;
    internal static BufferPool<double> StaticPool => ValueExtensions.DoublePool;
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
    public PreciseAcceleratedScalar(double value, int aidx) : base(StaticPool.Get(aidx, 1), []) => FromHost(value);
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
    public PreciseAcceleratedVector(double[] value, int aidx) : base(StaticPool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
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