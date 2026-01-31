using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public partial class LinearAlgebraKernels
{
    public static KernelStorage<Action<Index1D, TensorArrayView, float>> FillKernel { get; } = new(FillKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView,
        TensorArrayView>> ConcatKernel { get; } = new(ConcatKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, int, int>> SliceKernel { get; } = new(SliceKernelImpl);
    
    #region Elementwise Kernels
    #region Binary
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> AddKernel { get; } 
        = new(AddKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> SubtractKernel { get; } 
        = new(SubtractKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> ElementwiseMultiplyKernel { get; } 
        = new(ElementwiseMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> DivideKernel { get; } 
        = new(DivideKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> MaxKernel { get; } 
        = new(MaxKernelImpl);
    #endregion
    #region Unary
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> ExpKernel { get; } = new(ExpKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> LogKernel { get; } = new(LogKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> SqrtKernel { get; } = new(SqrtKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> AbsKernel { get; } = new(AbsKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> NegateKernel { get; } = new(NegateKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> SineKernel { get; } = new(SineKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> CosineKernel { get; } = new(CosineKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView>> TangentKernel { get; } = new(TangentKernelImpl);
    #endregion

    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> ScalarPowerKernel { get; } 
        = new(ScalarPowerKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> ScalarMultiplyKernel { get; } 
        = new(ScalarMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> ScalarDivideKernel { get; } 
        = new(ScalarDivideKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView>> ScalarMaxKernel { get; } 
        = new(ScalarMaxKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, float>> FloatPowerKernel { get; } = new(FloatPowerKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, float>> FloatMultiplyKernel { get; } = new(FloatMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, float>> FloatMaxKernel { get; } = new(FloatMaxKernelImpl);
    #endregion
    #region Matrix Kernels
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView, 
        int, int, int, int, float, float, int, int>> MatrixMultiplyKernel { get; } = new(MatrixMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView, 
        int, int, int, float, float, int>> MatrixVectorMultiplyKernel { get; } = new(MatrixVectorMultiplyKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, int, int>> TransposeKernel { get; } = new(TransposeKernelImpl);
    #endregion
    #region Vector Kernels
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, TensorArrayView, 
        int, int>> OuterProductKernel { get; } = new(OuterProductKernelImpl);
    public static KernelStorage<Action<Index1D, TensorArrayView, TensorArrayView, float>> AxpyKernel { get; } = new(AxpyKernelImpl);
    #endregion
}