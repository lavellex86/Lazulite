using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

#region TensorValue
public abstract class TensorValue<T>(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : Value<float, T>(buffer) where T : notnull
{
    public int[] Shape { get; } = shape;
    public abstract TensorProxy<T> ToProxy();
    
    public abstract TensorValue<T> Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape);
    public TensorValue<T> Zeros() => Create(Pool.GetLike(this), Shape);
    public TensorValue<T> Clone()
    {
        var buffer = Pool.GetLike(this, zero: false);
        buffer.CopyFrom(this);
        return Create(buffer, Shape);
    }
    public TensorValue<T> CreateAlike(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Create(buffer, Shape);

    public override BufferPool<float> Pool => StaticPool;
    internal static BufferPool<float> StaticPool => ValueExtensions.FloatPool;
}

public abstract class TensorProxy<T>(float[] flatData, int[] shape) where T : notnull
{
    public float[] FlatData { get; } = flatData;
    public int[] Shape { get; } = shape;
    
    protected TensorProxy(TensorValue<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract float Get(int[] index);
    public abstract T ToHost();

    public float this[int i] => FlatData[i];
    
    public static implicit operator float[](TensorProxy<T> proxy) => proxy.FlatData;
    public static implicit operator T(TensorProxy<T> proxy) => proxy.ToHost();
}
#endregion
#region ScalarValue
public class ScalarValue : TensorValue<float>
{
    public ScalarValue(float value, int aidx) : base(StaticPool.Get(aidx, 1), []) => FromHost(value);
    public ScalarValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, []) { }
    
    public override float Unroll(float[] rolled) => rolled[0];
    public override float[] Roll(float value) => [value];
    public override ScalarValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override ScalarProxy ToProxy() => new(this);
}

public class ScalarProxy(ScalarValue value) : TensorProxy<float>(value)
{
    public override float Get(int[] index) => FlatData[0];
    public override float ToHost() => FlatData[0];
}
#endregion
#region VectorValue
public class VectorValue : TensorValue<float[]>
{
    public VectorValue(float[] value, int aidx) : base(StaticPool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public VectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override VectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override VectorProxy ToProxy() => new(this);
}

public class VectorProxy(VectorValue value) : TensorProxy<float[]>(value)
{
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}
#endregion
#region MatrixValue
public class MatrixValue : TensorValue<float[,]>
{
    public MatrixValue(float[,] value, int aidx) : base(
        StaticPool.Get(aidx, value.GetLength(0) * value.GetLength(1)),
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public MatrixValue(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }
    
    public override float[] Roll(float[,] value) => MatrixProxy.Roll(value);
    public override float[,] Unroll(float[] rolled) => MatrixProxy.Unroll(rolled, Shape[1]);

    public override MatrixValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override MatrixProxy ToProxy() => new(this);
}

public class MatrixProxy(MatrixValue value) : TensorProxy<float[,]>(value)
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

#region PreciseTensorValue
public abstract class PreciseTensorValue<T>(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) : Value<double, T>(buffer) where T : notnull
{
    public int[] Shape { get; } = shape;
    public abstract PreciseTensorProxy<T> ToProxy();
    
    public abstract PreciseTensorValue<T> Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape);
    public PreciseTensorValue<T> Zeros() => Create(Pool.GetLike(this), Shape);
    public PreciseTensorValue<T> Clone()
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
    
    protected PreciseTensorProxy(PreciseTensorValue<T> data) : this(data.Data.View.GetAsArray1D(), data.Shape) { }

    public abstract double Get(int[] index);
    public abstract T ToHost();

    public double this[int i] => FlatData[i];
    
    public static implicit operator double[](PreciseTensorProxy<T> proxy) => proxy.FlatData;
    public static implicit operator T(PreciseTensorProxy<T> proxy) => proxy.ToHost();
}
#endregion
#region PreciseScalarValue
public class PreciseScalarValue : PreciseTensorValue<double>
{
    public PreciseScalarValue(double value, int aidx) : base(StaticPool.Get(aidx, 1), []) => FromHost(value);
    public PreciseScalarValue(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer, []) { }

    public override double Unroll(double[] rolled) => rolled[0];
    public override double[] Roll(double value) => [value];
    
    public override PreciseScalarValue Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) => new PreciseScalarValue(buffer);
    public override PreciseScalarProxy ToProxy() => new(this);
}

public class PreciseScalarProxy(PreciseScalarValue value) : PreciseTensorProxy<double>(value)
{
    public override double Get(int[] index) => FlatData[0];
    public override double ToHost() => FlatData[0];
}
#endregion
#region PreciseVectorValue
public class PreciseVectorValue : PreciseTensorValue<double[]>
{
    public PreciseVectorValue(double[] value, int aidx) : base(StaticPool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public PreciseVectorValue(MemoryBuffer1D<double, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }

    public override double[] Unroll(double[] rolled) => rolled;
    public override double[] Roll(double[] value) => value;
    
    public override PreciseVectorValue Create(MemoryBuffer1D<double, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override PreciseVectorProxy ToProxy() => new(this);
}

public class PreciseVectorProxy(PreciseVectorValue value) : PreciseTensorProxy<double[]>(value)
{
    public override double Get(int[] index) => FlatData[index[0]];
    public override double[] ToHost() => FlatData;
}
#endregion