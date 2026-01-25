using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

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

    public override BufferPool<float> Pool => Compute.FloatPool;
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
    public AcceleratedScalar(float value, int aidx) : base(Compute.FloatPool.Get(aidx, 1), []) => FromHost(value);
    public AcceleratedScalar(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, []) { }
    
    public override float Unroll(float[] rolled) => rolled[0];
    public override float[] Roll(float value) => [value];
    public override AcceleratedScalar Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override ScalarProxy ToProxy() => new(this);
    
    public static AcceleratedScalar operator +(AcceleratedScalar a, AcceleratedScalar b) => LinearAlgebraSuite.Add(a, b).AsScalar();
    public static AcceleratedScalar operator -(AcceleratedScalar a, AcceleratedScalar b) => LinearAlgebraSuite.Subtract(a, b).AsScalar();
    public static AcceleratedScalar operator *(AcceleratedScalar a, AcceleratedScalar b) => LinearAlgebraSuite.ElementwiseMultiply(a, b).AsScalar();
    public static AcceleratedScalar operator /(AcceleratedScalar a, AcceleratedScalar b) => LinearAlgebraSuite.Divide(a, b).AsScalar();
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
    public AcceleratedVector(float[] value, int aidx) : base(Compute.FloatPool.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public AcceleratedVector(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override AcceleratedVector Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override VectorProxy ToProxy() => new(this);
    
    public static AcceleratedVector operator +(AcceleratedVector a, AcceleratedVector b) => LinearAlgebraSuite.Add(a, b).AsVector();
    public static AcceleratedVector operator -(AcceleratedVector a, AcceleratedVector b) => LinearAlgebraSuite.Subtract(a, b).AsVector();
    public static AcceleratedVector operator *(AcceleratedVector a, AcceleratedVector b) => LinearAlgebraSuite.ElementwiseMultiply(a, b).AsVector();
    public static AcceleratedVector operator /(AcceleratedVector a, AcceleratedVector b) => LinearAlgebraSuite.Divide(a, b).AsVector();
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
        Compute.FloatPool.Get(aidx, value.GetLength(0) * value.GetLength(1)),
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public AcceleratedMatrix(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }
    
    public override float[] Roll(float[,] value) => MatrixProxy.Roll(value);
    public override float[,] Unroll(float[] rolled) => MatrixProxy.Unroll(rolled, Shape[1]);

    public override AcceleratedMatrix Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override MatrixProxy ToProxy() => new(this);
    
    public static AcceleratedMatrix operator +(AcceleratedMatrix a, AcceleratedMatrix b) => LinearAlgebraSuite.Add(a, b).AsMatrix();
    public static AcceleratedMatrix operator -(AcceleratedMatrix a, AcceleratedMatrix b) => LinearAlgebraSuite.Subtract(a, b).AsMatrix();
    public static AcceleratedMatrix operator *(AcceleratedMatrix a, AcceleratedMatrix b) => LinearAlgebraSuite.ElementwiseMultiply(a, b).AsMatrix();
    public static AcceleratedMatrix operator /(AcceleratedMatrix a, AcceleratedMatrix b) => LinearAlgebraSuite.Divide(a, b).AsMatrix();
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
        for (int j = 0; j < cols; j++) 
            vector[KernelProgramming.MatrixIndexOf(i, j, cols)] = value[i, j];
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