namespace Raphael.Lazulite;

public partial class Compute
{
    public static Value<,> Add<T>(Value<,> a, Value<,> b) where T : notnull => a.CreateAlike(Add(a.Data, b.Data));
    public static Value<,> Subtract<T>(Value<,> a, Value<,> b) where T : notnull => a.CreateAlike(Subtract(a.Data, b.Data));
    public static Value<,> ElementwiseMultiply<T>(Value<,> a, Value<,> b) where T : notnull => a.CreateAlike(ElementwiseMultiply(a.Data, b.Data));
    public static Value<,> Divide<T>(Value<,> a, Value<,> b) where T : notnull => a.CreateAlike(Divide(a.Data, b.Data));
    public static Value<,> Max<T>(Value<,> a, Value<,> b) where T : notnull => a.CreateAlike(Max(a.Data, b.Data));
    
    public static Value<,> Exp<T>(Value<,> val) where T : notnull => val.CreateAlike(Exp(val.Data));
    public static Value<,> Log<T>(Value<,> val) where T : notnull => val.CreateAlike(Log(val.Data));
    public static Value<,> Sqrt<T>(Value<,> val) where T : notnull => val.CreateAlike(Sqrt(val.Data));
    public static Value<,> Abs<T>(Value<,> val) where T : notnull => val.CreateAlike(Abs(val.Data));
    public static Value<,> Negate<T>(Value<,> val) where T : notnull => val.CreateAlike(Negate(val.Data));
    public static Value<,> Sine<T>(Value<,> val) where T : notnull => val.CreateAlike(Sine(val.Data));
    public static Value<,> Cosine<T>(Value<,> val) where T : notnull => val.CreateAlike(Cosine(val.Data));
    public static Value<,> Tangent<T>(Value<,> val) where T : notnull => val.CreateAlike(Tangent(val.Data));
    
    public static Value<,> ScalarPower<T>(Value<,> value, Value<,> scalar) where T : notnull => value.CreateAlike(ScalarPower(value.Data, scalar));
    public static Value<,> ScalarMultiply<T>(Value<,> value, Value<,> scalar) where T : notnull => value.CreateAlike(ScalarMultiply(value.Data, scalar));
    public static Value<,> ScalarDivide<T>(Value<,> value, Value<,> scalar) where T : notnull => value.CreateAlike(ScalarDivide(value.Data, scalar));
    public static Value<,> ScalarMax<T>(Value<,> value, Value<,> scalar) where T : notnull => value.CreateAlike(ScalarMax(value.Data, scalar));
    
    public static Value<,> FloatPower<T>(Value<,> value, float scalar) where T : notnull => value.CreateAlike(FloatPower(value.Data, scalar));
    public static Value<,> FloatMultiply<T>(Value<,> value, float scalar) where T : notnull => value.CreateAlike(FloatMultiply(value.Data, scalar));
    public static Value<,> FloatMax<T>(Value<,> value, float scalar) where T : notnull => value.CreateAlike(FloatMax(value.Data, scalar));
    
    public static Value<,> Sum<T>(Value<,> val) where T : notnull => new ScalarValue(Sum(val.Data));
    public static Value<,> Dot<T>(Value<,> a, Value<,> b) where T : notnull => new ScalarValue(Dot(a.Data, b.Data));
}