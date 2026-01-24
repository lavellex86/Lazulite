using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class TensorValue3 : Value<,>
{
    public TensorValue3(float[,,] value, int aidx) : base(
        Compute.Get(aidx, value.GetLength(0) * value.GetLength(1) * value.GetLength(2)),
        [value.GetLength(0), value.GetLength(1), value.GetLength(2)]) => FromHost(value);
    public TensorValue3(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) : base(buffer, shape) { }

    public override float[] Roll(float[,,] value) => TensorProxy3.Roll(value);
    public override float[,,] Unroll(float[] rolled) => TensorProxy3.Unroll(rolled, Shape[1], Shape[2]);
    public override TensorValue3 Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, shape);
    public override TensorProxy3 ToProxy() => new(this);
}

public class TensorProxy3(TensorValue3 value) : ValueProxy<float[,,]>(value)
{
    public float this[int i, int j, int k] => FlatData[IndexOf(i, j, k, Shape[1], Shape[2])];
    public override float Get(int[] index) => this[index[0], index[1], index[2]];
    public override float[,,] ToHost() => Unroll(FlatData, Shape[1], Shape[2]);

    public static float[,,] Unroll(float[] rolled, int d1, int d2)
    {
        var d0 = rolled.Length / (d1 * d2);
        var matrix = new float[d0, d1, d2];
        Parallel.For(0, d0, i =>
        {
            for (int j = 0; j < d1; j++)
            for (int k = 0; k < d2; k++)
                matrix[i, j, k] = rolled[IndexOf(i, j, k, d1, d2)];
        });
        return matrix;
    }
    public static float[] Roll(float[,,] value)
    {
        var (d0, d1, d2) = (value.GetLength(0), value.GetLength(1), value.GetLength(2));
        var vector = new float[d0 * d1 * d2];
        Parallel.For(0, d0, i =>
        {
            for (int j = 0; j < d1; j++)
            for (int k = 0; k < d2; k++)
                vector[IndexOf(i, j, k, d1, d2)] = value[i, j, k];
        });
        return vector;
    }
    
    public static int IndexOf(int x, int y, int z, int d1, int d2) 
        => x * d1 * d2 + y * d2 + z;
    public static (int x, int y, int z) FromIndex(int index, int d1, int d2) 
        => (index / (d1 * d2), index / d2 % d1, index % d2);
}