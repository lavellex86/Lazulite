using Raphael.Lazulite;

namespace Raphael.Linalg32;

public partial class Extensions
{
    public static RemoteTensor<T> Add<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.AddKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> Subtract<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SubtractKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> Multiply<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MultiplyKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> Divide<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.DivideKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> AddScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.AddScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> SubtractScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SubtractScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> MultiplyScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MultiplyScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> DivideScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.DivideScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> Exp<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.ExpKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Log<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.LogKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Log10<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.Log10Kernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Log2<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.Log2Kernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Sqrt<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SqrtKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Pow<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.PowKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> PowScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.PowScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> Sin<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SinKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Cos<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.CosKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Tan<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.TanKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Sinh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SinhKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Cosh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.CoshKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Tanh<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.TanhKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Abs<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.AbsKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Floor<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.FloorKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Ceiling<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.CeilingKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Round<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.RoundKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Truncate<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.TruncateKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Sign<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.SignKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Min<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MinKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> Max<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MaxKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> MinScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MinScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> MaxScalar<T>(this RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.MaxScalarKernel.Call(r!.IntLength, r, a, scalar));
    
    public static RemoteTensor<T> Negate<T>(this RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.NegateKernel.Call(r!.IntLength, r, a));
    
    public static RemoteTensor<T> Fill<T>(this RemoteTensor<T> tensor, float value) where T : notnull
    {
        tensor.Context.GetKernels().FillKernel.Call(tensor.IntLength, tensor, value);
        return tensor;
    }
    
    public static RemoteTensor<T> Concat<T>(this RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.ConcatKernel.Call(r!.IntLength, r, a, b));
    
    public static RemoteTensor<T> Slice<T>(this RemoteTensor<T> source, int start, int length, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, k => k.SliceKernel.Call(r!.IntLength, r, source, start, length));
    
    public static RemoteTensor<T> OuterProduct<T>(this RemoteTensor<T> a, RemoteTensor<T> b, int dimension, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.OuterProductKernel.Call(r!.IntLength, r, a, b, dimension));
    
    public static RemoteTensor<T> Axpy<T>(this RemoteTensor<T> a, float alpha, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, k => k.AxpyKernel.Call(r!.IntLength, r, a, alpha)); // check this later
    
    public static RemoteTensor<T> MatrixMultiply<T>(this RemoteTensor<T> a, RemoteTensor<T> b, int m, int n, float alpha = 1.0f, float beta = 0.0f, int k = 0, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(a, r, kernel => kernel.MatrixMultiplyKernel.Call(r!.IntLength, r, a, b, m, n, alpha, beta, k));
    
    public static RemoteTensor<T> Transpose<T>(this RemoteTensor<T> source, int dimension, RemoteTensor<T>? r = null) where T : notnull => 
        Encase(source, r, k => k.TransposeKernel.Call(r!.IntLength, r, source, dimension));
    
    private static RemoteTensor<T> Encase<T>(RemoteTensor<T> inferFrom, RemoteTensor<T>? r, Action<Kernels> action) where T : notnull
    {
        r ??= inferFrom.Create(inferFrom.Shape);
        action(inferFrom.Context.GetKernels());
        return r;
    }
}