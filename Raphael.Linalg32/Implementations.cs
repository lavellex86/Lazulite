using ILGPU;
using ILGPU.Algorithms;

namespace Raphael.Linalg32;

internal partial class Implementations
{
    internal static void Fill(Index1D i, FAV r, float n) => r[i] = n;
    internal static void Concat(Index1D i, FAV r, FAV a, FAV b) => r[i] = i < a.Length ? a[i] : b[i - a.Length];

    internal static void Slice(Index1D i, FAV r, FAV v, int start, int end)
    {
        if (i >= start && i < end) r[i - start] = v[i];
    }

    internal static void Add(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] + b[i];
    internal static void Subtract(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] - b[i];
    internal static void Multiply(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] * b[i];
    internal static void Divide(Index1D i, FAV r, FAV a, FAV b) => r[i] = a[i] / b[i];

    internal static void AddScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = a[i] + scalar;
    internal static void SubtractScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = a[i] - scalar;
    internal static void MultiplyScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = a[i] * scalar;
    internal static void DivideScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = a[i] / scalar;

    internal static void Exp(Index1D i, FAV r, FAV a) => r[i] = XMath.Exp(a[i]);
    internal static void Log(Index1D i, FAV r, FAV a) => r[i] = XMath.Log(a[i]);
    internal static void Sqrt(Index1D i, FAV r, FAV a) => r[i] = XMath.Sqrt(a[i]);
    internal static void Sin(Index1D i, FAV r, FAV a) => r[i] = XMath.Sin(a[i]);
    internal static void Cos(Index1D i, FAV r, FAV a) => r[i] = XMath.Cos(a[i]);
    internal static void Tan(Index1D i, FAV r, FAV a) => r[i] = XMath.Tan(a[i]);
    internal static void Sinh(Index1D i, FAV r, FAV a) => r[i] = XMath.Sinh(a[i]);
    internal static void Cosh(Index1D i, FAV r, FAV a) => r[i] = XMath.Cosh(a[i]);
    internal static void Tanh(Index1D i, FAV r, FAV a) => r[i] = XMath.Tanh(a[i]);

    internal static void Log10(Index1D i, FAV r, FAV a) => r[i] = XMath.Log10(a[i]);
    internal static void Log2(Index1D i, FAV r, FAV a) => r[i] = XMath.Log2(a[i]);
    internal static void Pow(Index1D i, FAV r, FAV a, FAV b) => r[i] = XMath.Pow(a[i], b[i]);
    internal static void PowScalar(Index1D i, FAV r, FAV a, float exponent) => r[i] = XMath.Pow(a[i], exponent);

    internal static void Abs(Index1D i, FAV r, FAV a) => r[i] = XMath.Abs(a[i]);
    internal static void Floor(Index1D i, FAV r, FAV a) => r[i] = XMath.Floor(a[i]);
    internal static void Ceiling(Index1D i, FAV r, FAV a) => r[i] = XMath.Ceiling(a[i]);
    internal static void Round(Index1D i, FAV r, FAV a) => r[i] = XMath.Round(a[i]);
    internal static void Truncate(Index1D i, FAV r, FAV a) => r[i] = XMath.Truncate(a[i]);
    internal static void Sign(Index1D i, FAV r, FAV a) => r[i] = XMath.Sign(a[i]);

    internal static void Min(Index1D i, FAV r, FAV a, FAV b) => r[i] = XMath.Min(a[i], b[i]);
    internal static void Max(Index1D i, FAV r, FAV a, FAV b) => r[i] = XMath.Max(a[i], b[i]);
    internal static void MinScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = XMath.Min(a[i], scalar);
    internal static void MaxScalar(Index1D i, FAV r, FAV a, float scalar) => r[i] = XMath.Max(a[i], scalar);

    internal static void Negate(Index1D i, FAV r, FAV a) => r[i] = -a[i];

    internal static void OuterProduct(Index1D i, FAV r, FAV a, FAV b, int n) => r[i] = a[i / n] * b[i % n];

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
        else if (transposeFlag == 3) (k, n) = (a0, b0); // transpose both
        else throw new Exception("MatrixMultiply got an invalid 'transposeFlag.'");

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
}