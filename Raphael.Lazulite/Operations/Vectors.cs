using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public partial class Compute
{
    public static void Sum(ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> a)
    {
        var aidx = a.AcceleratorIndex();
        Accelerators[aidx].Reduce<float, AddFloat>(
            GetStream(aidx),
            a, result);
    }
    

    public static void Dot(
        MemoryBuffer1D<float, Stride1D.Dense> result,
        MemoryBuffer1D<float, Stride1D.Dense> a, 
        MemoryBuffer1D<float, Stride1D.Dense> b, 
        bool noCuBlas = false)
    {
        var aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || result.Length < 1e3)
        {
            var temp = GetLike(a);
            Call(ElementwiseMultiplyKernel, temp, a, b);
            Sum(result, temp);
            Return(temp);
        }
        else
            blas.Dot(a.View.AsGeneral(), b.View.AsGeneral(), result.View.BaseView);
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b,
        bool noCuBlas = false) => 
        Encase(a.AcceleratorIndex(), 1, r => Dot(r, a, b, noCuBlas));

    public static void Axpy(
        float alpha,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> y,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || x.Length < 1e3) Call(AxpyKernel, x, y, alpha);
        else blas.Axpy(alpha, x.AsGeneral(), y.AsGeneral());
    }

    public static void Scale(
        float alpha,
        ArrayView1D<float, Stride1D.Dense> x,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || x.Length < 1e3)
            Call(FloatMultiplyKernel, x, x, alpha);
        else blas.Scal(alpha, x.AsGeneral());
    }

    public static void OuterProduct(
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> y,
        int m, int n, float alpha = 1.0f,
        bool noCuBlas = false) // x is m, y is n, result is m x n 
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || result.Length < 1e3)
            Call(OuterProductKernel, result, x, y, m, n);
        else
            blas.Ger(
                m, n, alpha,
                x.AsGeneral(),
                y.AsGeneral(),
                result.BaseView, m);
    }
    public static MemoryBuffer1D<float, Stride1D.Dense> OuterProduct(
        MemoryBuffer1D<float, Stride1D.Dense> x,
        MemoryBuffer1D<float, Stride1D.Dense> y,
        int m, int n, float alpha = 1.0f, bool noCuBlas = false) => 
        Encase(x.AcceleratorIndex(), m * n, r => OuterProduct(r, x, y, m, n, alpha, noCuBlas));

    public static MemoryBuffer1D<float, Stride1D.Dense> Concat(
        MemoryBuffer1D<float, Stride1D.Dense> a,
        MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Encase(a.AcceleratorIndex(), (int)(a.Length + b.Length), r => Call(ConcatKernel, r, a, b));
}