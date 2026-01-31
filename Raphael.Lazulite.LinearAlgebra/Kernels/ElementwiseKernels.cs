using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraKernels
{
    private static void FillKernelImpl(Index1D i, TensorArrayView view, float value) => view[i] = value;
    private static void ConcatKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = i < a.Length ? a[i] : b[i - a.Length];

    private static void SliceKernelImpl(Index1D i, TensorArrayView dest, TensorArrayView source, int start, int end)
    {
        if (i >= start && i < end) dest[i - start] = source[i];
    }
    
    #region Binary
    private static void AddKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] + b[i];
    private static void SubtractKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] - b[i];
    private static void ElementwiseMultiplyKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] * b[i];
    private static void DivideKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] / b[i];
    
    private static void MaxKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = XMath.Max(a[i], b[i]);
    #endregion
    #region Unary
    private static void ExpKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Exp(a[i]);
    private static void LogKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Log(a[i]);
    private static void SqrtKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Sqrt(a[i]);
    private static void AbsKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Abs(a[i]);
    private static void NegateKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = -a[i];
    private static void SineKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Sin(a[i]);
    private static void CosineKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Cos(a[i]);
    private static void TangentKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a) => result[i] = XMath.Tan(a[i]);
    #endregion
    #region Scalar & Float
    private static void ScalarPowerKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = XMath.Pow(a[i], b[0]);
    private static void ScalarMultiplyKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] * b[0];
    private static void ScalarDivideKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = a[i] / b[0];
    private static void ScalarMaxKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, TensorArrayView b) => result[i] = XMath.Max(a[i], b[0]);
    private static void FloatPowerKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, float power) => result[i] = XMath.Pow(a[i], power);
    private static void FloatMultiplyKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, float b) => result[i] = a[i] * b;
    private static void FloatMaxKernelImpl(Index1D i, TensorArrayView result, TensorArrayView a, float b) => result[i] = XMath.Max(a[i], b);
    #endregion
}