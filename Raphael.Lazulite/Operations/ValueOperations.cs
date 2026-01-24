namespace Raphael.Lazulite;

public partial class Compute
{
    public static AcceleratedValue<,> Add<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => a.CreateAlike(Add(a.Data, b.Data));
    public static AcceleratedValue<,> Subtract<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => a.CreateAlike(Subtract(a.Data, b.Data));
    public static AcceleratedValue<,> ElementwiseMultiply<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => a.CreateAlike(ElementwiseMultiply(a.Data, b.Data));
    public static AcceleratedValue<,> Divide<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => a.CreateAlike(Divide(a.Data, b.Data));
    public static AcceleratedValue<,> Max<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => a.CreateAlike(Max(a.Data, b.Data));
    
    public static AcceleratedValue<,> Exp<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Exp(val.Data));
    public static AcceleratedValue<,> Log<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Log(val.Data));
    public static AcceleratedValue<,> Sqrt<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Sqrt(val.Data));
    public static AcceleratedValue<,> Abs<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Abs(val.Data));
    public static AcceleratedValue<,> Negate<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Negate(val.Data));
    public static AcceleratedValue<,> Sine<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Sine(val.Data));
    public static AcceleratedValue<,> Cosine<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Cosine(val.Data));
    public static AcceleratedValue<,> Tangent<T>(AcceleratedValue<,> val) where T : notnull => val.CreateAlike(Tangent(val.Data));
    
    public static AcceleratedValue<,> ScalarPower<T>(AcceleratedValue<,> acceleratedValue, AcceleratedValue<,> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarPower(acceleratedValue.Data, scalar));
    public static AcceleratedValue<,> ScalarMultiply<T>(AcceleratedValue<,> acceleratedValue, AcceleratedValue<,> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarMultiply(acceleratedValue.Data, scalar));
    public static AcceleratedValue<,> ScalarDivide<T>(AcceleratedValue<,> acceleratedValue, AcceleratedValue<,> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarDivide(acceleratedValue.Data, scalar));
    public static AcceleratedValue<,> ScalarMax<T>(AcceleratedValue<,> acceleratedValue, AcceleratedValue<,> scalar) where T : notnull => acceleratedValue.CreateAlike(ScalarMax(acceleratedValue.Data, scalar));
    
    public static AcceleratedValue<,> FloatPower<T>(AcceleratedValue<,> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatPower(acceleratedValue.Data, scalar));
    public static AcceleratedValue<,> FloatMultiply<T>(AcceleratedValue<,> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatMultiply(acceleratedValue.Data, scalar));
    public static AcceleratedValue<,> FloatMax<T>(AcceleratedValue<,> acceleratedValue, float scalar) where T : notnull => acceleratedValue.CreateAlike(FloatMax(acceleratedValue.Data, scalar));
    
    public static AcceleratedValue<,> Sum<T>(AcceleratedValue<,> val) where T : notnull => new AcceleratedScalar(Sum(val.Data));
    public static AcceleratedValue<,> Dot<T>(AcceleratedValue<,> a, AcceleratedValue<,> b) where T : notnull => new AcceleratedScalar(Dot(a.Data, b.Data));
}