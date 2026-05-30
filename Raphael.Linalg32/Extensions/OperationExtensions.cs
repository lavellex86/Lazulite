using Raphael.Lazulite;

namespace Raphael.Linalg32;

public partial class Extensions
{
    public static RemoteTensor<T> Add<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AddKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Subtract<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SubtractKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Multiply<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MultiplyKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Divide<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.DivideKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> AddScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AddScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> SubtractScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SubtractScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> MultiplyScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MultiplyScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> DivideScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.DivideScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Exp<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.ExpKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.LogKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log10<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.Log10Kernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log2<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.Log2Kernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sqrt<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SqrtKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Pow<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.PowKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> PowScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.PowScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Sin<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SinKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Cos<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CosKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Tan<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TanKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sinh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SinhKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Cosh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CoshKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Tanh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TanhKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Abs<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AbsKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Floor<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.FloorKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Ceiling<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CeilingKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Round<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.RoundKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Truncate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TruncateKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sign<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SignKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Min<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MinKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Max<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MaxKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> MinScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MinScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> MaxScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MaxScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Negate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.NegateKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Fill<T>(this LazuliteContext _, RemoteTensor<T> tensor, float value) where T : notnull
    {
        tensor.Context.GetKernels().FillKernel.Call(tensor.IntLength, tensor, value);
        return tensor;
    }
    
    public static RemoteTensor<T> Concat<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.ConcatKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Slice<T>(this LazuliteContext _, RemoteTensor<T> source, int start, int length, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, (k, _r) => k.SliceKernel.Call(_r.IntLength, _r, source, start, length));

    public static RemoteTensor<float[,]> OuterProduct(this LazuliteContext ctx, RemoteTensor<float[]> a, RemoteTensor<float[]> b, RemoteTensor<float[,]>? r = null)
    {
        r ??= ctx.GetMatrix(a.IntLength, b.IntLength);
        ctx.GetKernels().OuterProductKernel.Call(r.IntLength, r, a, b);
        return r;
    }

    public static RemoteTensor<float[,]> MatrixMultiply(this LazuliteContext _, RemoteTensor<float[,]> a, RemoteTensor<float[,]> b, int m, int n, float alpha = 1.0f, float beta = 0.0f,
        RemoteTensor<float[,]>? r = null, bool transposeA = false, bool transposeB = false, bool useCuBlas = true)
    {
        var transposeFlag = (transposeA, transposeB) switch
        {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3
        };
        
        return Encase(a, r, (kernel, _r) => kernel.MatrixMultiplyKernel.Call(_r.IntLength, _r, a, b, m, n, alpha, beta, transposeFlag));
    }
    
    public static RemoteTensor<float[]> MatrixVectorMultiply(this LazuliteContext ctx, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, int m0, float alpha = 1.0f, float beta = 0.0f,
        RemoteTensor<float[]>? r = null, bool transposeM = false, bool useCuBlas = true)
    {
        var m1 = m.IntLength / m0;
        r ??= ctx.GetVector(transposeM ? m0 : m1);
        return ctx.MatrixMultiply(m, v.AsMatrix(), m0, 1, alpha, beta, r.AsMatrix(), transposeM, false, useCuBlas).AsVector();
    }
    
    public static RemoteTensor<T> Transpose<T>(this LazuliteContext _, RemoteTensor<T> source, int dimension, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, (k, _r) => k.TransposeKernel.Call(_r.IntLength, _r, source, dimension));

    public static RemoteTensor<float[,]> BroadcastMatrixVectorAdd(this LazuliteContext _, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, RemoteTensor<float[,]>? r = null) =>
        Encase(m, r, (k, _r) => k.BroadcastMatrixVectorAddKernel.Call(m.IntLength, _r, m, v));

    public static RemoteTensor<float[]> NarrowcastVectorMatrixAdd(this LazuliteContext _, RemoteTensor<float[]> v, RemoteTensor<float[,]> m, RemoteTensor<float[]>? r = null) =>
        Encase(v, r, (k, _r) => k.NarrowcastVectorMatrixAdd.Call(v.IntLength, _r, v, m, m.Shape[0]));
    
    private static RemoteTensor<T> Encase<T>(RemoteTensor<T> inferFrom, RemoteTensor<T>? r, Action<Kernels, RemoteTensor<T>> action) where T : notnull
    {
        r ??= inferFrom.Create(inferFrom.Shape);
        action(inferFrom.Context.GetKernels(), r);
        return r;
    }
}