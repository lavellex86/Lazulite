using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Raphael.Lazulite.LinearAlgebra;

public partial class LinearAlgebraSuite
{
    private readonly static Dictionary<int, CuBlas?> _cublasHandles = [];

    private static void InitializeCuBlas()
    {
        foreach (var aidx in Compute.Accelerators.Keys) GetCuBlas(aidx);
    }

    private static void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles.Values)
        {
            try { handle?.Dispose(); }
            catch { } // can't do anything about it now- this is at process exit
        }
        _cublasHandles.Clear();
    }

    private static CuBlas? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Compute.Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception) { _cublasHandles[aidx] = null; }
        return blas;
    }
    
    public static void Add(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Compute.Call(AddKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Add(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Add(r, a, b));

    public static void Subtract(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Compute.Call(SubtractKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Subtract(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Subtract(r, a, b));

    public static void ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Compute.Call(ElementwiseMultiplyKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        a.Encase(r => ElementwiseMultiply(r, a, b));

    public static void Divide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Compute.Call(DivideKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Divide(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Divide(r, a, b));

    public static void Max(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Compute.Call(MaxKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Max(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Max(r, a, b));

    public static void Exp(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(ExpKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Exp(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Exp(r, val));

    public static void Log(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(LogKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Log(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Log(r, val));

    public static void Sqrt(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(SqrtKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sqrt(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Sqrt(r, val));

    public static void Abs(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(AbsKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Abs(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Abs(r, val));

    public static void Negate(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(NegateKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Negate(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Negate(r, val));
    
    public static void Sine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(SineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Sine(r, val));
    
    public static void Cosine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(CosineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Cosine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Cosine(r, val));
    
    public static void Tangent(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Compute.Call(TangentKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Tangent(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Tangent(r, val));

    public static void ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarPowerKernel, r, value, scalar);

    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) =>
        value.Encase(r => ScalarPower(r, value, scalar));

    public static void ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarMultiply(r, value, scalar));

    public static void ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarDivideKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarDivide(r, value, scalar));

    public static void ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> value, MemoryBuffer1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarMax(r, value, scalar));
    
    public static void FloatPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatPowerKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatPower(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatPower(r, value, scalar));

    public static void FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatMultiply(r, value, scalar));

    public static void FloatMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMax(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatMax(r, value, scalar));

    public static MemoryBuffer1D<float, Stride1D.Dense> Sum(MemoryBuffer1D<float, Stride1D.Dense> val)
    {
        var result = val.Pool().Get(val.AcceleratorIndex(), 1);
        Sum(result, val);
        return result;
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b)
    {
        var result = a.Pool().Get(a.AcceleratorIndex(), 1);
        Dot(result, a, b);
        return result;
    }
}