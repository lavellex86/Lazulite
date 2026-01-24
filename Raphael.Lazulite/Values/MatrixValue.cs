using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class MatrixValue : Value<,>
{
    public MatrixValue(float[,] value, int aidx) : base(
        Compute.Get(aidx, value.GetLength(0) * value.GetLength(1)),
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public MatrixValue(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }
    
    public override float[] Roll(float[,] value) => MatrixProxy.Roll(value);
    public override float[,] Unroll(float[] rolled) => MatrixProxy.Unroll(rolled, Shape[1]);

    public override MatrixValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override MatrixProxy ToProxy() => new(this);

    public static MatrixValue operator +(MatrixValue a, MatrixValue b) => Compute.Add(a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a, MatrixValue b) => Compute.Subtract(a, b).AsMatrix();
    public static MatrixValue operator *(MatrixValue a, MatrixValue b) => Compute.ElementwiseMultiply(a, b).AsMatrix();
    public static MatrixValue operator /(MatrixValue a, MatrixValue b) => Compute.Divide(a, b).AsMatrix();
    public static MatrixValue operator -(MatrixValue a) => Compute.Negate(a).AsMatrix();
}

public class MatrixProxy(Value<,> value) : ValueProxy<float[,]>(value)
{
    public float this[int i, int j] => FlatData[KernelProgramming.MatrixIndexOf(i, j, Shape[1])];
    public override float Get(int[] index) => this[index[0], index[1]];
    public override float[,] ToHost() => Unroll(FlatData, Shape[1]);
    
    public static float[] Roll(float[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        var vector = new float[rows * cols];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                vector[KernelProgramming.MatrixIndexOf(i, j, cols)] = value[i, j];
        });
        return vector;
    }
    
    public static float[,] Unroll(float[] rolled, int cols)
    {
        var rows = rolled.Length / cols;
        var matrix = new float[rows, cols];
        Parallel.For(0, rows, i =>
        {
            for (int j = 0; j < cols; j++)
                matrix[i, j] = rolled[KernelProgramming.MatrixIndexOf(i, j, cols)];
        });
        return matrix;
    }
    
    
}