using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public partial class LinearAlgebraKernels
{
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>> FillKernel { get; } = new(FillKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ConcatKernel { get; } = new(ConcatKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> SliceKernel { get; } = new(SliceKernelImpl);
    
    #region Elementwise Kernels
    #region Binary
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AddKernel { get; } 
        = new(AddKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SubtractKernel { get; } 
        = new(SubtractKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseMultiplyKernel { get; } 
        = new(ElementwiseMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> DivideKernel { get; } 
        = new(DivideKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> MaxKernel { get; } 
        = new(MaxKernelImpl);
    #endregion
    #region Unary
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ExpKernel { get; } = new(ExpKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> LogKernel { get; } = new(LogKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SqrtKernel { get; } = new(SqrtKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AbsKernel { get; } = new(AbsKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> NegateKernel { get; } = new(NegateKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SineKernel { get; } = new(SineKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CosineKernel { get; } = new(CosineKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> TangentKernel { get; } = new(TangentKernelImpl);
    #endregion

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarPowerKernel { get; } 
        = new(ScalarPowerKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarMultiplyKernel { get; } 
        = new(ScalarMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarDivideKernel { get; } 
        = new(ScalarDivideKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarMaxKernel { get; } 
        = new(ScalarMaxKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatPowerKernel { get; } = new(FloatPowerKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMultiplyKernel { get; } = new(FloatMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMaxKernel { get; } = new(FloatMaxKernelImpl);
    #endregion
    #region Matrix Kernels
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int, int, int, float, float, int, int>> MatrixMultiplyKernel { get; } = new(MatrixMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int, float, float, int>> MatrixVectorMultiplyKernel { get; } = new(MatrixVectorMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> TransposeKernel { get; } = new(TransposeKernelImpl);
    #endregion
    #region Vector Kernels
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int>> OuterProductKernel { get; } = new(OuterProductKernelImpl);
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> AxpyKernel { get; } = new(AxpyKernelImpl);
    #endregion
}