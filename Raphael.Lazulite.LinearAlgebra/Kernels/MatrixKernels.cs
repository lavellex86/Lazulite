using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraKernels
{
    private static void MatrixMultiplyKernelImpl(
        Index1D i,
        TensorArrayView result,
        TensorArrayView a,
        TensorArrayView b,
        int a0, int a1, int b0, int b1,
        float alpha, float beta,
        int transposeA, int transposeB)
    {
        var m = transposeA == 1 ? a1 : a0;
        var k = transposeA == 1 ? a0 : a1;
        var n = transposeB == 1 ? b0 : b1;
        var (row, col) = (i / n, i % n);

        if (row >= m) return;

        var sum = 0f;
        for (int j = 0; j < k; j++)
        {
            var aIndex = transposeA == 1 ? j * a1 + row : row * a1 + j;
            var bIndex = transposeB == 1 ? col * b1 + j : j * b1 + col;
            sum += a[aIndex] * b[bIndex];
        }

        var resultIdx = row * n + col;
        result[resultIdx] = alpha * sum + beta * result[resultIdx];
    }

    private static void MatrixVectorMultiplyKernelImpl(
        Index1D i,
        TensorArrayView result,
        TensorArrayView matrix,
        TensorArrayView vector,
        int m0, int m1, int v0,
        float alpha, float beta, int transposeMatrix)
    {
        if (i >= m0) return;
        
        if (transposeMatrix == 1)
        {
            var sum = 0f;
            for (int row = 0; row < m1; row++) sum += matrix[row * m0 + i] * vector[row];
            result[i] = alpha * sum + beta * result[i];
        }
        else
        {
            var sum = 0f;
            for (int col = 0; col < m1; col++) sum += matrix[i * m1 + col] * vector[col];
            result[i] = alpha * sum + beta * result[i];
        }
    }

    private static void TransposeKernelImpl(Index1D i, TensorArrayView result, TensorArrayView matrix, int m, int n)
    {
        if (i >= m * n) return;
        var row = i / n;
        var col = i % n;
        result[col * m + row] = matrix[row * n + col];
    }
}