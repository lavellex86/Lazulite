using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Raphael.Lazulite.Suite;

public static partial class LinearAlgebra
{
    #region Binary
    public static void AddKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] + b[index];
    
    public static void SubtractKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] - b[index];
    
    public static void ElementwiseMultiplyKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[index];
    
    public static void DivideKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[index];
    
    public static void MaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[index]);
    #endregion
    #region Unary
    public static void ExpKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Exp(a[index]);

    public static void LogKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Log(a[index]);
    
    public static void SqrtKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sqrt(a[index]);
    
    public static void AbsKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Abs(a[index]);
    
    public static void NegateKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = -a[index];
    
    public static void SineKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sin(a[index]);
    
    public static void CosineKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Cos(a[index]);
    
    public static void TangentKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Tan(a[index]);
    #endregion
    #region Weird Ones
    public static void ScalarPowerKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Pow(a[index], b[0]);

    public static void ScalarMultiplyKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[0];
    
    public static void ScalarDivideKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[0];
    
    public static void ScalarMaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[0]);
    
    public static void FloatPowerKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float power) =>
        result[index] = XMath.Pow(a[index], power);
    
    public static void FloatMultiplyKernelImpl(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float b) =>
        result[index] = a[index] * b;

    public static void FloatMaxKernelImpl(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, float b) =>
        result[index] = XMath.Max(a[index], b);
    #endregion
}