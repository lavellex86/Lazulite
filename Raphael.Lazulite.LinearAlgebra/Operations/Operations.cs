using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using static Raphael.Lazulite.LinearAlgebra.LinearAlgebraKernels;

namespace Raphael.Lazulite.LinearAlgebra;

public partial class LinearAlgebraSuite
{
    private readonly static Dictionary<int, CuBlas<CuBlasPointerModeHandlers.AutomaticMode>?> _cublasHandles = [];

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

    private static CuBlas<CuBlasPointerModeHandlers.AutomaticMode>? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Compute.Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas<CuBlasPointerModeHandlers.AutomaticMode>(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception) { _cublasHandles[aidx] = null; }
        return blas;
    }
    
    public static void Fill(TensorArrayView r, float val) => Compute.Call(FillKernel, r, val);
    public static void Concat(TensorArrayView r, TensorArrayView a, TensorArrayView b) => 
        Compute.Call(ConcatKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Concat(TensorArrayView a, TensorArrayView b) 
    {
        var result = Compute.FloatPool.Get(a.AcceleratorIndex(), (int)(a.Length + b.Length));
        Compute.Call(ConcatKernel, result, a, b);
        return result;
    }
    public static void Slice(TensorArrayView r, TensorArrayView val, int start, int end) => 
        Compute.Call(SliceKernel, r, val, start, end); 
    public static MemoryBuffer1D<float, Stride1D.Dense> Slice(TensorArrayView val, int start, int end)
    {
        var result = Compute.FloatPool.Get(val.AcceleratorIndex(), end - start);
        Compute.Call(SliceKernel, result, val, start, end);
        return result;
    }
    
    public static void Add(TensorArrayView r, TensorArrayView a, TensorArrayView b) =>
        Compute.Call(AddKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Add(TensorArrayView a, TensorArrayView b) => 
        a.Encase(r => Add(r, a, b));

    public static void Subtract(TensorArrayView r, TensorArrayView a, TensorArrayView b) =>
        Compute.Call(SubtractKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Subtract(TensorArrayView a, TensorArrayView b) => 
        a.Encase(r => Subtract(r, a, b));

    public static void ElementwiseMultiply(TensorArrayView r, TensorArrayView a, TensorArrayView b) =>
        Compute.Call(ElementwiseMultiplyKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(TensorArrayView a, TensorArrayView b) => 
        a.Encase(r => ElementwiseMultiply(r, a, b));

    public static void Divide(TensorArrayView r, TensorArrayView a, TensorArrayView b) =>
        Compute.Call(DivideKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Divide(TensorArrayView a, TensorArrayView b) => 
        a.Encase(r => Divide(r, a, b));

    public static void Max(TensorArrayView r, TensorArrayView a, TensorArrayView b) =>
        Compute.Call(MaxKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Max(TensorArrayView a, TensorArrayView b) => 
        a.Encase(r => Max(r, a, b));

    public static void Exp(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(ExpKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Exp(TensorArrayView val) => 
        val.Encase(r => Exp(r, val));

    public static void Log(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(LogKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Log(TensorArrayView val) => 
        val.Encase(r => Log(r, val));

    public static void Sqrt(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(SqrtKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sqrt(TensorArrayView val) => 
        val.Encase(r => Sqrt(r, val));

    public static void Abs(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(AbsKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Abs(TensorArrayView val) => 
        val.Encase(r => Abs(r, val));

    public static void Negate(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(NegateKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Negate(TensorArrayView val) => 
        val.Encase(r => Negate(r, val));
    
    public static void Sine(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(SineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sine(TensorArrayView val) =>
        val.Encase(r => Sine(r, val));
    
    public static void Cosine(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(CosineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Cosine(TensorArrayView val) =>
        val.Encase(r => Cosine(r, val));
    
    public static void Tangent(TensorArrayView r, TensorArrayView val) =>
        Compute.Call(TangentKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Tangent(TensorArrayView val) =>
        val.Encase(r => Tangent(r, val));

    public static void ScalarPower(TensorArrayView r, TensorArrayView value, TensorArrayView scalar) =>
        Compute.Call(ScalarPowerKernel, r, value, scalar);

    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(TensorArrayView value, TensorArrayView scalar) =>
        value.Encase(r => ScalarPower(r, value, scalar));

    public static void ScalarMultiply(TensorArrayView r, TensorArrayView value, TensorArrayView scalar) =>
        Compute.Call(ScalarMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(TensorArrayView value, TensorArrayView scalar) => 
        value.Encase(r => ScalarMultiply(r, value, scalar));

    public static void ScalarDivide(TensorArrayView r, TensorArrayView value, TensorArrayView scalar) =>
        Compute.Call(ScalarDivideKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(TensorArrayView value, TensorArrayView scalar) => 
        value.Encase(r => ScalarDivide(r, value, scalar));

    public static void ScalarMax(TensorArrayView r, TensorArrayView value, TensorArrayView scalar) =>
        Compute.Call(ScalarMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(TensorArrayView value, TensorArrayView scalar) => 
        value.Encase(r => ScalarMax(r, value, scalar));
    
    public static void FloatPower(TensorArrayView r, TensorArrayView value, float scalar) =>
        Compute.Call(FloatPowerKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatPower(TensorArrayView value, float scalar) => 
        value.Encase(r => FloatPower(r, value, scalar));

    public static void FloatMultiply(TensorArrayView r, TensorArrayView value, float scalar) =>
        Compute.Call(FloatMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(TensorArrayView value, float scalar) => 
        value.Encase(r => FloatMultiply(r, value, scalar));

    public static void FloatMax(TensorArrayView r, TensorArrayView value, float scalar) =>
        Compute.Call(FloatMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMax(TensorArrayView value, float scalar) => 
        value.Encase(r => FloatMax(r, value, scalar));

    public static MemoryBuffer1D<float, Stride1D.Dense> Sum(TensorArrayView val)
    {
        var result = Compute.FloatPool.Get(val.AcceleratorIndex(), 1);
        Sum(result, val);
        return result;
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(TensorArrayView a, TensorArrayView b)
    {
        var result = Compute.FloatPool.Get(a.AcceleratorIndex(), 1);
        Dot(result, a, b);
        return result;
    }
}