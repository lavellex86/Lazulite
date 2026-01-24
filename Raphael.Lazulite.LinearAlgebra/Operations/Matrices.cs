using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Raphael.Lazulite.Suite;

public partial class LinearAlgebra
{
    public static void MatrixMultiply(
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        int a0, int a1, int b0, int b1,
        float alpha = 1.0f, float beta = 0.0f,
        bool transposeA = false, bool transposeB = false,
        bool noCuBlas = false)
    {
        int aidx = a.AcceleratorIndex();
        var blas = GetCuBlas(aidx);
        if (blas is null || noCuBlas || result.Length < 1e3) 
            Compute.Call(MatrixMultiplyKernel, result, a, b, a0, a1, b0, b1, alpha, beta, transposeA ? 1 : 0, transposeB ? 1 : 0);
        else
        {
            int m = transposeA ? a1 : a0;
            int k = transposeA ? a0 : a1;
            int n = transposeB ? b1 : b0;
            
            blas.Gemm(
                transposeB ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose,
                transposeA ? CuBlasOperation.NonTranspose : CuBlasOperation.Transpose,
                n, m, k,
                alpha,
                b.BaseView, b1,
                a.BaseView, a1,
                beta,
                result.BaseView, n);}
    }

    public static void MatrixVectorMultiply(
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> matrix,
        ArrayView1D<float, Stride1D.Dense> vector,
        int m, int n, float alpha = 1.0f, float beta = 0.0f,
        bool transposeMatrix = false, bool noCuBlas = false) // matrix is m x n, vector is n, result is m
    {
        var aidx = matrix.AcceleratorIndex();
        var blas = GetCuBlas(aidx);

        if (blas is null || noCuBlas || matrix.Length < 1e3)
            Compute.Call(MatrixVectorMultiplyKernel, result, matrix, vector, m, n, alpha, beta, transposeMatrix ? 1 : 0);
        else
            blas.Gemv(
            transposeMatrix ? CuBlasOperation.Transpose : CuBlasOperation.NonTranspose,
            n, m, alpha,
            matrix.BaseView, n,
            vector.AsGeneral(), beta,
            result.AsGeneral());
    }

    public static void Transpose(
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> matrix,
        int m, int n) => Compute.Call(TransposeKernel, result, matrix, m, n);
}