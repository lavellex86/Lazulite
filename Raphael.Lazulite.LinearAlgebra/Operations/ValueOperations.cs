namespace Raphael.Lazulite.Suite;

public partial class LinearAlgebra
{
    public static AcceleratedTensor<T> Add<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => a.CreateAlike(Add(a.Data, b.Data));
    public static AcceleratedTensor<T> Subtract<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => a.CreateAlike(Subtract(a.Data, b.Data));
    public static AcceleratedTensor<T> ElementwiseMultiply<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => a.CreateAlike(ElementwiseMultiply(a.Data, b.Data));
    public static AcceleratedTensor<T> Divide<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => a.CreateAlike(Divide(a.Data, b.Data));
    public static AcceleratedTensor<T> Max<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => a.CreateAlike(Max(a.Data, b.Data));
    
    public static AcceleratedTensor<T> Exp<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Exp(val.Data));
    public static AcceleratedTensor<T> Log<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Log(val.Data));
    public static AcceleratedTensor<T> Sqrt<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Sqrt(val.Data));
    public static AcceleratedTensor<T> Abs<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Abs(val.Data));
    public static AcceleratedTensor<T> Negate<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Negate(val.Data));
    public static AcceleratedTensor<T> Sine<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Sine(val.Data));
    public static AcceleratedTensor<T> Cosine<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Cosine(val.Data));
    public static AcceleratedTensor<T> Tangent<T>(AcceleratedTensor<T> val) where T : notnull => val.CreateAlike(Tangent(val.Data));
    
    public static AcceleratedTensor<T> ScalarPower<T>(AcceleratedTensor<T> acceleratedValue, AcceleratedTensor<T> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarPower(acceleratedValue.Data, scalar));
    public static AcceleratedTensor<T> ScalarMultiply<T>(AcceleratedTensor<T> acceleratedValue, AcceleratedTensor<T> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarMultiply(acceleratedValue.Data, scalar));
    public static AcceleratedTensor<T> ScalarDivide<T>(AcceleratedTensor<T> acceleratedValue, AcceleratedTensor<T> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarDivide(acceleratedValue.Data, scalar));
    public static AcceleratedTensor<T> ScalarMax<T>(AcceleratedTensor<T> acceleratedValue, AcceleratedTensor<T> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarMax(acceleratedValue.Data, scalar));
    
    public static AcceleratedTensor<T> FloatPower<T>(AcceleratedTensor<T> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatPower(acceleratedValue.Data, scalar));
    public static AcceleratedTensor<T> FloatMultiply<T>(AcceleratedTensor<T> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatMultiply(acceleratedValue.Data, scalar));
    public static AcceleratedTensor<T> FloatMax<T>(AcceleratedTensor<T> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatMax(acceleratedValue.Data, scalar));
    
    public static AcceleratedScalar Sum<T>(AcceleratedTensor<T> val) where T : notnull => new(Sum(val.Data));
    public static AcceleratedScalar Dot<T>(AcceleratedTensor<T> a, AcceleratedTensor<T> b) where T : notnull => new(Dot(a.Data, b.Data));
}