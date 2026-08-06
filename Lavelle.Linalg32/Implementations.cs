using System.Diagnostics.CodeAnalysis;
using ILGPU;
using ILGPU.Algorithms;

namespace Lavelle.Linalg32;

internal partial class Implementations
{
    internal static void Fill(Index1D i, FAV r, float n) => r[i] = n;
    internal static void Concat(Index1D i, FAV r, FAV a, FAV b) => r[i] = i < a.Length ? a[i] : b[i - a.Length];

    internal static void Slice(Index1D i, FAV r, FAV v, int start) => r[i] = v[i + start];

    internal static void Add(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] + b[i];
    internal static void Subtract(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] - b[i];
    internal static void Multiply(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] * b[i];
    internal static void Divide(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] / b[i];

    internal static void AddScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = x[i] + scalar;
    internal static void SubtractScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = x[i] - scalar;
    internal static void MultiplyScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = x[i] * scalar;
    internal static void DivideScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = x[i] / scalar;

    internal static void Exp(Index1D i, FAV r, FAV x) => r[i] = XMath.Exp(x[i]);
    internal static void Log(Index1D i, FAV r, FAV x) => r[i] = XMath.Log(x[i]);
    internal static void Sqrt(Index1D i, FAV r, FAV x) => r[i] = XMath.Sqrt(x[i]);
    internal static void Sin(Index1D i, FAV r, FAV x) => r[i] = XMath.Sin(x[i]);
    internal static void Cos(Index1D i, FAV r, FAV x) => r[i] = XMath.Cos(x[i]);
    internal static void Tan(Index1D i, FAV r, FAV x) => r[i] = XMath.Tan(x[i]);
    internal static void Sinh(Index1D i, FAV r, FAV x) => r[i] = XMath.Sinh(x[i]);
    internal static void Cosh(Index1D i, FAV r, FAV x) => r[i] = XMath.Cosh(x[i]);
    internal static void Tanh(Index1D i, FAV r, FAV x) => r[i] = XMath.Tanh(x[i]);

    internal static void Log10(Index1D i, FAV r, FAV x) => r[i] = XMath.Log10(x[i]);
    internal static void Log2(Index1D i, FAV r, FAV x) => r[i] = XMath.Log2(x[i]);
    internal static void Pow(Index1D i, FAV r, FAV x, FAV b) => r[i] = XMath.Pow(x[i], b[i]);
    internal static void PowScalar(Index1D i, FAV r, FAV x, float exponent) => r[i] = XMath.Pow(x[i], exponent);

    internal static void Abs(Index1D i, FAV r, FAV x) => r[i] = XMath.Abs(x[i]);
    internal static void Floor(Index1D i, FAV r, FAV x) => r[i] = XMath.Floor(x[i]);
    internal static void Ceiling(Index1D i, FAV r, FAV x) => r[i] = XMath.Ceiling(x[i]);
    internal static void Round(Index1D i, FAV r, FAV x) => r[i] = XMath.Round(x[i]);
    internal static void Truncate(Index1D i, FAV r, FAV x) => r[i] = XMath.Truncate(x[i]);
    internal static void Sign(Index1D i, FAV r, FAV x) => r[i] = XMath.Sign(x[i]);

    internal static void Min(Index1D i, FAV r, FAV x, FAV b) => r[i] = XMath.Min(x[i], b[i]);
    internal static void Max(Index1D i, FAV r, FAV x, FAV b) => r[i] = XMath.Max(x[i], b[i]);
    internal static void MinScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = XMath.Min(x[i], scalar);
    internal static void MaxScalar(Index1D i, FAV r, FAV x, float scalar) => r[i] = XMath.Max(x[i], scalar);

    internal static void Negate(Index1D i, FAV r, FAV x) => r[i] = -x[i];

    public static void OuterProduct(Index1D index, FAV r, FAV a, FAV b, float alpha, float beta)
    {
        var (i, j) = (index % b.IntLength, index / b.IntLength);
        r[index] = alpha * a[i] * b[j] + beta * r[index];
    }

    internal static void MatrixMultiply(Index1D i, FAV r, FAV a, FAV b, int a0, int b0, float alpha, float beta, int transposeFlag)
    {
        var (a1, b1) = (a.IntLength / a0, b.IntLength / b0);
        var (transposeA, transposeB) = (false, false);
        if (transposeFlag == 1 || transposeFlag == 3) transposeA = true;
        if (transposeFlag == 2 || transposeFlag == 3) transposeB = true;

        int k, n;
        if (transposeFlag == 0) (k, n) = (a1, b1); // no transpose
        else if (transposeFlag == 1) (k, n) = (a0, b1); // transpose A
        else if (transposeFlag == 2) (k, n) = (a1, b0); // transpose B
        else (k, n) = (a0, b0); // transpose both
        var (row, col) = (i / n, i % n);

        var sum = 0f;
        for (int j = 0; j < k; j++)
        {
            var aIndex = transposeA ? j * a1 + row : row * a1 + j;
            var bIndex = transposeB ? col * b1 + j : j * b1 + col;
            sum += a[aIndex] * b[bIndex];
        }

        var resultIndex = row * n + col;
        r[resultIndex] = alpha * sum + beta * r[resultIndex];
    }

    internal static void Transpose(Index1D i, FAV r, FAV m, int a0)
    {
        var a1 = m.IntLength / a0;
        var (row, col) = (i / a0, i % a0);
        r[col * a1 + row] = m[row * a0 + col];
    }

    internal static void BroadcastMatrixVectorAdd(Index1D i, FAV r, FAV m, FAV v) => r[i] += v[i % v.IntLength];

    internal static void NarrowcastVectorMatrixAdd(Index1D i, FAV r, FAV m, int m0)
    {
        var (sum, m1) = (0f, m.IntLength / m0);
        for (int row = 0; row < m0; row++) sum += m[row * m1 + (int)i];
        r[i] += sum;
    }

    internal static void Dot(Index1D i, FAV r, FAV a, FAV b) => Atomic.Add(ref r[0], a[i] * b[i]);
    internal static void Axpy(Index1D i, FAV r, FAV a, FAV b, float alpha) => r[i] = alpha * a[i] + b[i];

    internal static void L1Norm(Index1D i, FAV r, FAV v) => Atomic.Add(ref r[0], XMath.Abs(v[i]));
    internal static void L2Norm(Index1D i, FAV r, FAV v) => Atomic.Add(ref v[0], v[i] * v[i]);

    internal static void Clamp(Index1D i, FAV r, FAV x, float min, float max) => r[i] = XMath.Clamp(x[i], min, max);
    internal static void Reciprocal(Index1D i, FAV r, FAV x) => r[i] = 1f / x[i];
    internal static void Atan2(Index1D i, FAV r, FAV y, FAV x) => r[i] = XMath.Atan2(y[i], x[i]);
    internal static void Sum(Index1D i, FAV r, FAV x) => Atomic.Add(ref r[0], x[i]);
}