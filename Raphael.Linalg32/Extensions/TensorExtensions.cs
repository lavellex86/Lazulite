using Raphael.Lazulite;

namespace Raphael.Linalg32;

public partial class Extensions
{
    public static RemoteTensor<T> Add<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AddKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Subtract<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SubtractKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Multiply<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MultiplyKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Divide<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.DivideKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> AddScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AddScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> SubtractScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SubtractScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> MultiplyScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MultiplyScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> DivideScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.DivideScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Exp<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.ExpKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.LogKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log10<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.Log10Kernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Log2<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.Log2Kernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sqrt<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SqrtKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Pow<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.PowKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> PowScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.PowScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Sin<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SinKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Cos<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CosKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Tan<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TanKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sinh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SinhKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Cosh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CoshKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Tanh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TanhKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Abs<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AbsKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Floor<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.FloorKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Ceiling<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.CeilingKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Round<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.RoundKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Truncate<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.TruncateKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Sign<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.SignKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Min<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MinKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Max<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MaxKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> MinScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MinScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> MaxScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.MaxScalarKernel.Call(_r.IntLength, _r, a, scalar));
    
    public static RemoteTensor<T> Negate<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.NegateKernel.Call(_r.IntLength, _r, a));
    
    public static RemoteTensor<T> Fill<T>(this RemoteTensor<T> tensor, float value) where T : notnull
    {
        tensor.Context.GetKernels().FillKernel.Call(tensor.IntLength, tensor, value);
        return tensor;
    }
    
    public static RemoteTensor<T> Concat<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.ConcatKernel.Call(_r.IntLength, _r, a, b));
    
    public static RemoteTensor<T> Slice<T>(this RemoteTensor<T> source, int start, int length, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, (k, _r) => k.SliceKernel.Call(_r.IntLength, _r, source, start, length));
    
    public static RemoteTensor<T> OuterProduct<T>(this RemoteTensor<T> a, RemoteTensor<T> b, int dimension, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.OuterProductKernel.Call(_r.IntLength, _r, a, b, dimension));
    
    public static RemoteTensor<T> Axpy<T>(this RemoteTensor<T> a, float alpha, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, (k, _r) => k.AxpyKernel.Call(_r.IntLength, _r, a, alpha)); // TODO: check this later

    public static RemoteTensor<T> MatrixMultiply<T>(this RemoteTensor<T> a, RemoteTensor<T> b, int m, int n, float alpha = 1.0f, float beta = 0.0f,
        RemoteTensor<T>? r = null, bool transposeA = false, bool transposeB = false, bool useCuBlas = true)
        where T : notnull
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
    
    public static RemoteTensor<T> Transpose<T>(this RemoteTensor<T> source, int dimension, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, (k, _r) => k.TransposeKernel.Call(_r.IntLength, _r, source, dimension));
    
    private static RemoteTensor<T> Encase<T>(RemoteTensor<T> inferFrom, RemoteTensor<T>? r, Action<Kernels, RemoteTensor<T>> action) where T : notnull
    {
        r ??= inferFrom.Create(inferFrom.Shape);
        action(inferFrom.Context.GetKernels(), r);
        return r;
    }
}