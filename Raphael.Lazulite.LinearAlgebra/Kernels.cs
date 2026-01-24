using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public partial class LinearAlgebraSuite
{
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, float>> FillKernel { get; } = new(FillKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ConcatKernel { get; } = new(ConcatKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> SliceKernel { get; } = new(SliceKernelImpl);
    
    #region Elementwise Kernels
    #region Binary
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AddKernel { get; } 
        = new(AddKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SubtractKernel { get; } 
        = new(SubtractKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ElementwiseMultiplyKernel { get; } 
        = new(ElementwiseMultiplyKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> DivideKernel { get; } 
        = new(DivideKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> MaxKernel { get; } 
        = new(MaxKernelImpl);
    #endregion
    #region Unary
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ExpKernel { get; } = new(ExpKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> LogKernel { get; } = new(LogKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SqrtKernel { get; } = new(SqrtKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AbsKernel { get; } = new(AbsKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> NegateKernel { get; } = new(NegateKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SineKernel { get; } = new(SineKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CosineKernel { get; } = new(CosineKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> TangentKernel { get; } = new(TangentKernelImpl);
    #endregion

    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarPowerKernel { get; } 
        = new(ScalarPowerKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarMultiplyKernel { get; } 
        = new(ScalarMultiplyKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarDivideKernel { get; } 
        = new(ScalarDivideKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ScalarMaxKernel { get; } 
        = new(ScalarMaxKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatPowerKernel { get; } = new(FloatPowerKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMultiplyKernel { get; } = new(FloatMultiplyKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMaxKernel { get; } = new(FloatMaxKernelImpl);
    #endregion
    #region Matrix Kernels
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int, int, int, float, float, int, int>> MatrixMultiplyKernel { get; } = new(MatrixMultiplyKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int, float, float, int>> MatrixVectorMultiplyKernel { get; } = new(MatrixVectorMultiplyKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> TransposeKernel { get; } = new(TransposeKernelImpl);
    #endregion
    #region Vector Kernels
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, 
        int, int>> OuterProductKernel { get; } = new(OuterProductKernelImpl);
    private static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> AxpyKernel { get; } = new(AxpyKernelImpl);
    #endregion
}