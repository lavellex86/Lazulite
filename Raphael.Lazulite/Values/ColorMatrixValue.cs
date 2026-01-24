using System.Drawing;
using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class ColorMatrixValue : Value<,>
{
    public ColorMatrixValue(Color[,] value, int aidx) : base(
        Compute.Get(aidx, value.GetLength(0) * value.GetLength(1)), 
        [value.GetLength(0), value.GetLength(1)]) => FromHost(value);
    public ColorMatrixValue(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }

    public override float[] Roll(Color[,] value) => ColorMatrixProxy.Roll(value);
    public override Color[,] Unroll(float[] rolled) => ColorMatrixProxy.Unroll(rolled, Shape[1]);
    public override ColorMatrixValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override ColorMatrixProxy ToProxy() => new(this);
}

public class ColorMatrixProxy(ColorMatrixValue value) : ValueProxy<Color[,]>(value)
{
    public float this[int i, int j] => FlatData[KernelProgramming.MatrixIndexOf(i, j, Shape[1])];
    public override float Get(int[] index) => this[index[0], index[1]];
    public override Color[,] ToHost() => Unroll(FlatData, Shape[1]);

    public static float[] Roll(Color[,] value)
    {
        var (rows, cols) = (value.GetLength(0), value.GetLength(1));
        var vector = new float[rows * cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                vector[KernelProgramming.MatrixIndexOf(i, j, cols)] = ColorVectorProxy.ColorToFloat(value[i, j]);
        }
        return vector;
    }

    public static Color[,] Unroll(float[] rolled, int cols)
    {
        var rows = rolled.Length / cols;
        var matrix = new Color[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
                matrix[i, j] = ColorVectorProxy.FloatToColor(rolled[KernelProgramming.MatrixIndexOf(i, j, cols)]);
        }
        return matrix;
    }
}