using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace Raphael.Lazulite;

public partial class Compute
{
    private readonly static Dictionary<int, CuBlas?> _cublasHandles = [];

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> AxpyKernel { get; private set; } =
        new((i, x, y, alpha) => y[i] = alpha * x[i] + y[i]);

    public static void InitializeCuBlas()
    {
        foreach (var aidx in Accelerators.Keys) GetCuBlas(aidx);
    }

    public static void CleanupCuBlas()
    {
        foreach (var handle in _cublasHandles.Values) handle?.Dispose();
        _cublasHandles.Clear();
    }

    public static CuBlas? GetCuBlas(int aidx)
    {
        if (_cublasHandles.TryGetValue(aidx, out var blas) || Accelerators[aidx] is not CudaAccelerator cudaAccelerator) return blas;
        try
        {
            blas = new CuBlas(cudaAccelerator);
            _cublasHandles[aidx] = blas;
        }
        catch (Exception) { _cublasHandles[aidx] = null; }
        return blas;
    }
    
    public static void Fill(MemoryBuffer1D<float, Stride1D.Dense> buffer, float value) => Call(FillKernel, buffer, value);
    public static void Zero(MemoryBuffer1D<float, Stride1D.Dense> buffer) => Call(ZeroKernel, buffer);
    public static void Copy(MemoryBuffer1D<float, Stride1D.Dense> dest, MemoryBuffer1D<float, Stride1D.Dense> src) => Call(CopyKernel, dest, src);

    public static void Add(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(AddKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Add(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Add(r, a, b));

    public static void Subtract(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(SubtractKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Subtract(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Subtract(r, a, b));

    public static void ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(ElementwiseMultiplyKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> ElementwiseMultiply(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => ElementwiseMultiply(r, a, b));

    public static void Divide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(DivideKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Divide(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Divide(r, a, b));

    public static void Max(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Call(MaxKernel, r, a, b);
    public static MemoryBuffer1D<float, Stride1D.Dense> Max(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) => 
        Encase(a, r => Max(r, a, b));

    public static void Exp(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(ExpKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Exp(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Exp(r, val));

    public static void Log(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(LogKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Log(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Log(r, val));

    public static void Sqrt(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(SqrtKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sqrt(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Sqrt(r, val));

    public static void Abs(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(AbsKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Abs(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Abs(r, val));

    public static void Negate(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(NegateKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Negate(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val, r => Negate(r, val));
    
    public static void Sine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(SineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Sine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Sine(r, val));
    
    public static void Cosine(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(CosineKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Cosine(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Cosine(r, val));
    
    public static void Tangent(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Call(TangentKernel, r, val);
    public static MemoryBuffer1D<float, Stride1D.Dense> Tangent(MemoryBuffer1D<float, Stride1D.Dense> val) =>
        Encase(val, r => Tangent(r, val));

    public static void ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) =>
        Call(ScalarPowerKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarPower(MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) => 
        Encase(value, r => ScalarPower(r, value, scalar));

    public static void ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) =>
        Call(ScalarMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) => 
        Encase(value, r => ScalarMultiply(r, value, scalar));

    public static void ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) =>
        Call(ScalarDivideKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarDivide(MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) => 
        Encase(value, r => ScalarDivide(r, value, scalar));

    public static void ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) =>
        Call(ScalarMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> ScalarMax(MemoryBuffer1D<float, Stride1D.Dense> value, AcceleratedValue<,> scalar) => 
        Encase(value, r => ScalarMax(r, value, scalar));
    
    public static void FloatPower(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatPowerKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatPower(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatPower(r, value, scalar));

    public static void FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMultiplyKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMultiply(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMultiply(r, value, scalar));

    public static void FloatMax(MemoryBuffer1D<float, Stride1D.Dense> r, MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) =>
        Call(FloatMaxKernel, r, value, scalar);
    public static MemoryBuffer1D<float, Stride1D.Dense> FloatMax(MemoryBuffer1D<float, Stride1D.Dense> value, float scalar) => 
        Encase(value, r => FloatMax(r, value, scalar));

    public static MemoryBuffer1D<float, Stride1D.Dense> Sum(MemoryBuffer1D<float, Stride1D.Dense> val) => 
        Encase(val.AcceleratorIndex(), 1, r => Sum(r, val));
    
    public static MemoryBuffer1D<float, Stride1D.Dense> Dot(MemoryBuffer1D<float, Stride1D.Dense> a, MemoryBuffer1D<float, Stride1D.Dense> b) =>
        Encase(a.AcceleratorIndex(), 1, r => Dot(r, a, b));
}