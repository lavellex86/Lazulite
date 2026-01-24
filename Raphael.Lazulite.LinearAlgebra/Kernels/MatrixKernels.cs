using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraSuite
{
    public static void MatrixMultiplyKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b,
        int a0, int a1, int b0, int b1,
        float alpha, float beta,
        int transposeA, int transposeB)
    {
        int m = transposeA == 1 ? a1 : a0;
        int k = transposeA == 1 ? a0 : a1;
        int n = transposeB == 1 ? b0 : b1;
        var (row, col) = (index / n, index % n);

        if (row >= m) return;

        float sum = 0;
        for (int i = 0; i < k; i++)
        {
            int aIdx = transposeA == 1 ? i * a1 + row : row * a1 + i;
            int bIdx = transposeB == 1 ? col * b1 + i : i * b1 + col;
            sum += a[aIdx] * b[bIdx];
        }

        int resultIdx = row * n + col;
        result[resultIdx] = alpha * sum + beta * result[resultIdx];
    }

    public static void MatrixVectorMultiplyKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> matrix,
        ArrayView1D<float, Stride1D.Dense> vector,
        int m, int n,
        float alpha, float beta, int transposeMatrix)
    {
        if (transposeMatrix == 1)
        {
            int col = index.X;
            if (col >= n) return;

            float sum = 0;
            for (int row = 0; row < m; row++) sum += matrix[row * n + col] * vector[row];
            result[col] = alpha * sum + beta * result[col];
        }
        else
        {
            int row = index.X;
            if (row >= m) return;

            float sum = 0;
            for (int col = 0; col < n; col++) sum += matrix[row * n + col] * vector[col];
            result[row] = alpha * sum + beta * result[row];
        }
    }

    public static void TransposeKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> matrix,
        int m, int n)
    {
        if (index >= m * n) return;
    
        int row = index / n;
        int col = index % n;
    
        result[col * m + row] = matrix[row * n + col];
    }
}