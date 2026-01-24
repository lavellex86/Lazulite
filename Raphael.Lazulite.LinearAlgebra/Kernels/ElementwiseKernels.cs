using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Raphael.Lazulite.Kernels;

public static class ElementwiseKernels
{
    #region Binary
    public static void AddKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] + b[index];
    
    public static void SubtractKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] - b[index];
    
    public static void ElementwiseMultiplyKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[index];
    
    public static void DivideKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[index];
    
    public static void MaxKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[index]);
    #endregion
    
    #region Unary
    public static void ExpKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Exp(a[index]);

    public static void LogKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Log(a[index]);
    
    public static void SqrtKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sqrt(a[index]);
    
    public static void AbsKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Abs(a[index]);
    
    public static void NegateKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = -a[index];
    
    public static void SineKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Sin(a[index]);
    
    public static void CosineKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Cos(a[index]);
    
    public static void TangentKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a) =>
        result[index] = XMath.Tan(a[index]);
    #endregion

    #region Weird Ones
    public static void ScalarPowerKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Pow(a[index], b[0]);

    public static void ScalarMultiplyKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] * b[0];
    
    public static void ScalarDivideKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = a[index] / b[0];
    
    public static void ScalarMaxKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a,
        ArrayView1D<float, Stride1D.Dense> b) =>
        result[index] = XMath.Max(a[index], b[0]);
    
    public static void FloatPowerKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float power) =>
        result[index] = XMath.Pow(a[index], power);
    
    public static void FloatMultiplyKernel(
        Index1D index,
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, 
        float b) =>
        result[index] = a[index] * b;

    public static void FloatMaxKernel(
        Index1D index, 
        ArrayView1D<float, Stride1D.Dense> result,
        ArrayView1D<float, Stride1D.Dense> a, float b) =>
        result[index] = XMath.Max(a[index], b);
    #endregion
}