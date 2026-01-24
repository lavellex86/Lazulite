using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;

namespace Raphael.Lazulite.Suite;

public partial class LinearAlgebra
{
    public static void Sum(ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> a)
    {
        var aidx = a.AcceleratorIndex();
        Compute.Accelerators[aidx].Reduce<float, AddFloat>(Compute.GetStream(aidx), a, result);
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
            var temp = a.Pool().GetLike(a);
            Compute.Call(ElementwiseMultiplyKernel, temp, a, b);
            Sum(result, temp);
            temp.Return();
        }
        else
            blas.Dot(a.View.AsGeneral(), b.View.AsGeneral(), result.View.BaseView);
    }

    public static void Axpy(
        float alpha,
        ArrayView1D<float, Stride1D.Dense> x,
        ArrayView1D<float, Stride1D.Dense> y,
        bool noCuBlas = false)
    {
        var aidx = x.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || x.Length < 1e3) Compute.Call(AxpyKernel, x, y, alpha);
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
            Compute.Call(FloatMultiplyKernel, x, x, alpha);
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
            Compute.Call(OuterProductKernel, result, x, y, m, n);
        else
            blas.Ger(
                m, n, alpha,
                x.AsGeneral(),
                y.AsGeneral(),
                result.BaseView, m);
    }
}