using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraKernels
{
    private static void FillKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> view, float value) => view[index] = value;
    private static void ConcatKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> result, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
    {
        if (index < a.Length)
            result[index] = a[index];
        else
            result[index] = b[index - a.Length];
    }
    private static void SliceKernelImpl(Index1D index, ArrayView1D<float, Stride1D.Dense> dest, ArrayView1D<float, Stride1D.Dense> source, int start, int end)
    {
        if (index >= start && index < end) dest[index - start] = source[index];
    }
    
    #region Binary
    private static void AddKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] + b[index];
    
    private static void SubtractKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] - b[index];
    
    private static void ElementwiseMultiplyKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[index];
    
    private static void DivideKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[index];
    
    private static void MaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[index]);
    #endregion
    #region Unary
    private static void ExpKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Exp(a[index]);

    private static void LogKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Log(a[index]);
    
    private static void SqrtKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sqrt(a[index]);
    
    private static void AbsKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Abs(a[index]);
    
    private static void NegateKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = -a[index];
    
    private static void SineKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sin(a[index]);
    
    private static void CosineKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Cos(a[index]);
    
    private static void TangentKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Tan(a[index]);
    #endregion
    #region Weird Ones
    private static void ScalarPowerKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Pow(a[index], b[0]);

    private static void ScalarMultiplyKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[0];
    
    private static void ScalarDivideKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[0];
    
    private static void ScalarMaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[0]);
    
    private static void FloatPowerKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float power) =>
        result[index] = XMath.Pow(a[index], power);
    
    private static void FloatMultiplyKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float b) =>
        result[index] = a[index] * b;

    private static void FloatMaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, float b) =>
        result[index] = XMath.Max(a[index], b);
    #endregion
}