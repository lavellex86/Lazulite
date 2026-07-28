using Raphael.Lazulite;

namespace Raphael.Linalg32;

public partial class LinalgExtensions
{
    /// <summary>
    /// Adds two tensors elementwise.
    /// </summary>
    public static RemoteTensor<T> Add<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.AddKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Subtracts two tensors elementwise.
    /// </summary>
    public static RemoteTensor<T> Subtract<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SubtractKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Multiplies two tensors elementwise.
    /// </summary>
    public static RemoteTensor<T> Multiply<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MultiplyKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Divides two tensors elementwise.
    /// </summary>
    public static RemoteTensor<T> Divide<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.DivideKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Adds a scalar value to every element of a tensor.
    /// </summary>
    public static RemoteTensor<T> AddScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.AddScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Subtracts a scalar value from every element of a tensor.
    /// </summary>
    public static RemoteTensor<T> SubtractScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SubtractScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Multiplies every element of a tensor by a scalar value.
    /// </summary>
    public static RemoteTensor<T> MultiplyScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MultiplyScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Divides every element of a tensor by a scalar value.
    /// </summary>
    public static RemoteTensor<T> DivideScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.DivideScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Computes the elementwise natural exponential (e^x) of a tensor.
    /// </summary>
    public static RemoteTensor<T> Exp<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.ExpKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise natural logarithm of a tensor.
    /// </summary>
    public static RemoteTensor<T> Log<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.LogKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise base-10 logarithm of a tensor.
    /// </summary>
    public static RemoteTensor<T> Log10<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.Log10Kernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise base-2 logarithm of a tensor.
    /// </summary>
    public static RemoteTensor<T> Log2<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.Log2Kernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise square root of a tensor.
    /// </summary>
    public static RemoteTensor<T> Sqrt<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SqrtKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Raises each element of <paramref name="a"/> to the power of the corresponding element of <paramref name="b"/>.
    /// </summary>
    public static RemoteTensor<T> Pow<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.PowKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Raises every element of a tensor to a scalar power.
    /// </summary>
    public static RemoteTensor<T> PowScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.PowScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Computes the elementwise sine of a tensor.
    /// </summary>
    public static RemoteTensor<T> Sin<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SinKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise cosine of a tensor.
    /// </summary>
    public static RemoteTensor<T> Cos<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.CosKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise tangent of a tensor.
    /// </summary>
    public static RemoteTensor<T> Tan<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.TanKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise hyperbolic sine of a tensor.
    /// </summary>
    public static RemoteTensor<T> Sinh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SinhKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise hyperbolic cosine of a tensor.
    /// </summary>
    public static RemoteTensor<T> Cosh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.CoshKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise hyperbolic tangent of a tensor.
    /// </summary>
    public static RemoteTensor<T> Tanh<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.TanhKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise absolute value of a tensor.
    /// </summary>
    public static RemoteTensor<T> Abs<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.AbsKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise floor of a tensor, rounding each element down to the nearest integer.
    /// </summary>
    public static RemoteTensor<T> Floor<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.FloorKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise ceiling of a tensor, rounding each element up to the nearest integer.
    /// </summary>
    public static RemoteTensor<T> Ceiling<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.CeilingKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Rounds each element of a tensor to the nearest integer.
    /// </summary>
    public static RemoteTensor<T> Round<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.RoundKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Truncates each element of a tensor toward zero, discarding any fractional part.
    /// </summary>
    public static RemoteTensor<T> Truncate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.TruncateKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Returns the elementwise sign of a tensor (-1, 0, or 1 for each element).
    /// </summary>
    public static RemoteTensor<T> Sign<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.SignKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Computes the elementwise minimum of two tensors.
    /// </summary>
    public static RemoteTensor<T> Min<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MinKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Computes the elementwise maximum of two tensors.
    /// </summary>
    public static RemoteTensor<T> Max<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T> b, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MaxKernel.Call(_r.Length, _r, a, b));

    /// <summary>
    /// Clamps each element of a tensor to a maximum of <paramref name="scalar"/>, returning the elementwise minimum.
    /// </summary>
    public static RemoteTensor<T> MinScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MinScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Clamps each element of a tensor to a minimum of <paramref name="scalar"/>, returning the elementwise maximum.
    /// </summary>
    public static RemoteTensor<T> MaxScalar<T>(this LazuliteContext _, RemoteTensor<T> a, float scalar, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.MaxScalarKernel.Call(_r.Length, _r, a, scalar));

    /// <summary>
    /// Negates every element of a tensor.
    /// </summary>
    public static RemoteTensor<T> Negate<T>(this LazuliteContext _, RemoteTensor<T> a, RemoteTensor<T>? r = null) where T : notnull =>
        Encase(a, r, (k, _r) => k.NegateKernel.Call(_r.Length, _r, a));

    /// <summary>
    /// Fills every element of a tensor in-place with the given scalar value and returns it.
    /// </summary>
    public static RemoteTensor<T> Fill<T>(this LazuliteContext lctx, RemoteTensor<T> tensor, float value) where T : notnull
    {
        lctx.GetKernels().FillKernel.Call(tensor.Length, tensor, value);
        return tensor;
    }

    /// <summary>
    /// Concatenates two vectors.
    /// </summary>
    public static RemoteVector Concat<T>(this LazuliteContext lctx, RemoteVector a, RemoteVector b, RemoteVector? r = null) where T : notnull
    {
        r ??= lctx.GetVector(a.Length + b.Length);
        lctx.GetKernels().ConcatKernel.Call(r.Length, r, a, b);
        return r;
    }

    /// <summary>
    /// Extracts a contiguous slice of a vector starting at the given index.
    /// </summary>
    public static RemoteVector Slice<T>(this LazuliteContext lctx, RemoteVector source, int start, int end, RemoteVector? r = null) where T : notnull
    {
        r ??= lctx.GetVector(end - start);
        lctx.GetKernels().SliceKernel.Call(r.Length, r, source, start);
        return r;
    }

    /// <summary>
    /// Computes the outer product of two vectors, producing a matrix of shape [a.Length, b.Length].
    /// </summary>
    public static RemoteTensor<float[,]> OuterProduct(this LazuliteContext ctx, RemoteTensor<float[]> a, RemoteTensor<float[]> b, RemoteTensor<float[,]>? r = null)
    {
        r ??= ctx.GetMatrix(a.Length, b.Length);
        ctx.GetKernels().OuterProductKernel.Call(r.Length, r, a, b);
        return r;
    }

    /// <summary>
    /// Multiplies two matrices, optionally transposing either input. Supports scaling the product by
    /// <paramref name="alpha"/> and accumulating into <paramref name="r"/> scaled by <paramref name="beta"/>.
    /// </summary>
    public static RemoteTensor<float[,]> MatrixMultiply(this LazuliteContext ctx, RemoteTensor<float[,]> a, RemoteTensor<float[,]> b,
        float alpha = 1.0f, float beta = 0.0f, RemoteTensor<float[,]>? r = null,
        bool transposeA = false, bool transposeB = false, bool useCuBlas = false)
    {
        var aRows = transposeA ? a.Shape[1] : a.Shape[0];
        var aCols = transposeA ? a.Shape[0] : a.Shape[1];
        var bCols = transposeB ? b.Shape[0] : b.Shape[1];
        r ??= ctx.GetMatrix(aRows, bCols);
        var a0 = a.Shape[0];
        var b0 = b.Shape[0];
        var transposeFlag = (transposeA, transposeB) switch
        {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3
        };
        ctx.GetKernels().MatrixMultiplyKernel.Call(r.Length, r, a, b, a0, b0, alpha, beta, transposeFlag);
        return r;
    }

    /// <summary>
    /// Multiplies a matrix by a vector, optionally transposing the matrix. Supports scaling via
    /// <paramref name="alpha"/> and <paramref name="beta"/> in the same manner as <see cref="MatrixMultiply"/>.
    /// </summary>
    public static RemoteTensor<float[]> MatrixVectorMultiply(this LazuliteContext ctx, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, int m0, float alpha = 1.0f, float beta = 0.0f,
        RemoteTensor<float[]>? r = null, bool transposeM = false, bool useCuBlas = true)
    {
        var m1 = m.Length / m0;
        r ??= ctx.GetVector(transposeM ? m0 : m1);
        return ctx.MatrixMultiply(m, v.AsMatrix(), alpha, beta, r.AsMatrix(), transposeM, false, useCuBlas).AsVector();
    }

    /// <summary>
    /// Transposes a tensor along the specified dimension.
    /// </summary>
    public static RemoteTensor<float[,]> Transpose(this LazuliteContext ctx, RemoteTensor<float[,]> source, RemoteTensor<float[,]>? r = null)
    {
        r ??= ctx.GetMatrix(source.Shape[1], source.Shape[0]);
        ctx.GetKernels().TransposeKernel.Call(r.Length, r, source, source.Shape[0]);
        return r;
    }

    /// <summary>
    /// Adds a vector to every row of a matrix, broadcasting the vector across the matrix's first dimension.
    /// </summary>
    public static RemoteTensor<float[,]> BroadcastMatrixVectorAdd(this LazuliteContext _, RemoteTensor<float[,]> m, RemoteTensor<float[]> v, RemoteTensor<float[,]>? r = null) =>
        Encase(m, r, (k, _r) => k.BroadcastMatrixVectorAddKernel.Call(m.Length, _r, m, v));

    /// <summary>
    /// Reduces a matrix into a vector by summing each column across rows, then adds the result to <paramref name="v"/>.
    /// </summary>
    public static RemoteTensor<float[]> NarrowcastVectorMatrixAdd(this LazuliteContext _, RemoteTensor<float[]> v, RemoteTensor<float[,]> m, RemoteTensor<float[]>? r = null) =>
        Encase(v, r, (k, _r) => k.NarrowcastVectorMatrixAdd.Call(v.Length, _r, v, m, m.Shape[0]));

    private static RemoteTensor<T> Encase<T>(RemoteTensor<T> inferFrom, RemoteTensor<T>? r, Action<Kernels, RemoteTensor<T>> action) where T : notnull
    {
        r ??= inferFrom.Create(inferFrom.Shape);
        action(inferFrom.Context.GetKernels(), r);
        return r;
    }
}