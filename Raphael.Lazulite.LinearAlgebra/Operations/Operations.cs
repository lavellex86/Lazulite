using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using static Raphael.Lazulite.LinearAlgebra.LinearAlgebraKernels;

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
    
    public static void Fill(ArrayView1D<float, Stride1D.Dense> r, float val) => Compute.Call(FillKernel, r, val);
    public static void Concat(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        Compute.Call(ConcatKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Concat(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) 
    {
        var result = Compute.FloatPool.Get(a.AcceleratorIndex(), (int)(a.Length + b.Length));
        Compute.Call(ConcatKernel, result, a, b);
        return result;
    }
    public static void Slice(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val, int start, int end) => 
        Compute.Call(SliceKernel, r, val, start, end); 
    public static MemoryBuffer1D<float, Stride1D.Dense> Slice(ArrayView1D<float, Stride1D.Dense> val, int start, int end)
    {
        var result = Compute.FloatPool.Get(val.AcceleratorIndex(), end - start);
        Compute.Call(SliceKernel, result, val, start, end);
        return result;
    }
    
    public static void Add(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) =>
        Compute.Call(AddKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Add(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Add(r, a, b));

    public static void Subtract(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) =>
        Compute.Call(SubtractKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Subtract(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Subtract(r, a, b));

    public static void ElementwiseMultiply(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) =>
        Compute.Call(ElementwiseMultiplyKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        a.Encase(r => ElementwiseMultiply(r, a, b));

    public static void Divide(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) =>
        Compute.Call(DivideKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Divide(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Divide(r, a, b));

    public static void Max(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) =>
        Compute.Call(MaxKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Max(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) => 
        a.Encase(r => Max(r, a, b));

    public static void Exp(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(ExpKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Exp(ArrayView1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Exp(r, val));

    public static void Log(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(LogKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Log(ArrayView1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Log(r, val));

    public static void Sqrt(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(SqrtKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sqrt(ArrayView1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Sqrt(r, val));

    public static void Abs(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(AbsKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Abs(ArrayView1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Abs(r, val));

    public static void Negate(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(NegateKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Negate(ArrayView1D<float, Stride1D.Dense> val) => 
        val.Encase(r => Negate(r, val));
    
    public static void Sine(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(SineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sine(ArrayView1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Sine(r, val));
    
    public static void Cosine(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(CosineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Cosine(ArrayView1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Cosine(r, val));
    
    public static void Tangent(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> val) =>
        Compute.Call(TangentKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Tangent(ArrayView1D<float, Stride1D.Dense> val) =>
        val.Encase(r => Tangent(r, val));

    public static void ScalarPower(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarPowerKernel, r, value, scalar);

    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) =>
        value.Encase(r => ScalarPower(r, value, scalar));

    public static void ScalarMultiply(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarMultiply(r, value, scalar));

    public static void ScalarDivide(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarDivideKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarDivide(r, value, scalar));

    public static void ScalarMax(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) =>
        Compute.Call(ScalarMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(ArrayView1D<float, Stride1D.Dense> value, ArrayView1D<float, Stride1D.Dense> scalar) => 
        value.Encase(r => ScalarMax(r, value, scalar));
    
    public static void FloatPower(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatPowerKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatPower(ArrayView1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatPower(r, value, scalar));

    public static void FloatMultiply(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(ArrayView1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatMultiply(r, value, scalar));

    public static void FloatMax(ArrayView1D<float, Stride1D.Dense> r, ArrayView1D<float, Stride1D.Dense> value, float scalar) =>
        Compute.Call(FloatMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMax(ArrayView1D<float, Stride1D.Dense> value, float scalar) => 
        value.Encase(r => FloatMax(r, value, scalar));

    public static MemoryBuffer1D<float, Stride1D.Dense> Sum(ArrayView1D<float, Stride1D.Dense> val)
    {
        var result = Compute.FloatPool.Get(val.AcceleratorIndex(), 1);
        Sum(result, val);
        return result;
    }

    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b)
    {
        var result = Compute.FloatPool.Get(a.AcceleratorIndex(), 1);
        Dot(result, a, b);
        return result;
    }
}