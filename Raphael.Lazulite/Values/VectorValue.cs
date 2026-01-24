using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class VectorValue : Value<,>
{
    public VectorValue(float[] value, int aidx) : base(Compute.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public VectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }
    
    public override float[] Unroll(float[] rolled) => rolled;
    public override float[] Roll(float[] value) => value;
    public override VectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override VectorProxy ToProxy() => new(this);

    public static VectorValue operator +(VectorValue a, VectorValue b) => Compute.Add(a, b).AsVector();
    public static VectorValue operator -(VectorValue a, VectorValue b) => Compute.Subtract(a, b).AsVector();
    public static VectorValue operator *(VectorValue a, VectorValue b) => Compute.ElementwiseMultiply(a, b).AsVector();
    public static VectorValue operator *(VectorValue a, ScalarValue b) => Compute.ScalarMultiply(a, b).AsVector();
    public static VectorValue operator /(VectorValue a, VectorValue b) => Compute.Divide(a, b).AsVector();
    public static VectorValue operator /(VectorValue a, ScalarValue b) => Compute.ScalarDivide(a, b).AsVector();
    public static VectorValue operator -(VectorValue a) => Compute.Negate(a).AsVector();

    public ScalarValue Sum() => Compute.Sum(this).AsScalar();
    public ScalarValue Dot(VectorValue b) => Compute.Dot(this, b).AsScalar();
}

public class VectorProxy(VectorValue value) : ValueProxy<float[]>(value)
{
    public override float Get(int[] index) => FlatData[index[0]];
    public override float[] ToHost() => FlatData;
}