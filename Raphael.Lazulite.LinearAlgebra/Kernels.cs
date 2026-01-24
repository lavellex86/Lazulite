namespace Raphael.Lazulite.Suite;

public static partial class Kernels
{
    #region Elementwise Kernels
    #region Binary
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> AddKernel { get; private set; } = new(ElementwiseKernels.AddKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> SubtractKernel { get; private set; } = new(ElementwiseKernels.SubtractKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ElementwiseMultiplyKernel { get; private set; } = new(ElementwiseKernels.ElementwiseMultiplyKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> DivideKernel { get; private set; } = new(ElementwiseKernels.DivideKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> MaxKernel { get; private set; } = new(ElementwiseKernels.MaxKernel);
    #endregion
    #region Unary
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> ExpKernel { get; private set; } =
        new(ElementwiseKernels.ExpKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> LogKernel { get; private set; } =
        new(ElementwiseKernels.LogKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SqrtKernel { get; private set; } =
        new(ElementwiseKernels.SqrtKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> AbsKernel { get; private set; } =
        new(ElementwiseKernels.AbsKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> NegateKernel { get; private set; } =
        new(ElementwiseKernels.NegateKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> SineKernel { get; private set; } =
        new(ElementwiseKernels.SineKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> CosineKernel { get; private set; } =
        new(ElementwiseKernels.CosineKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>> TangentKernel { get; private set; } =
        new(ElementwiseKernels.TangentKernel);
    #endregion

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ScalarPowerKernel { get; private set; } = new(ElementwiseKernels.ScalarPowerKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ScalarMultiplyKernel { get; private set; } = new(ElementwiseKernels.ScalarMultiplyKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ScalarDivideKernel { get; private set; } = new(ElementwiseKernels.ScalarDivideKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>>> ScalarMaxKernel { get; private set; } = new(ElementwiseKernels.ScalarMaxKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatPowerKernel =
        new(ElementwiseKernels.FloatPowerKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMultiplyKernel =
        new(ElementwiseKernels.FloatMultiplyKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, float>> FloatMaxKernel =
        new(ElementwiseKernels.FloatMaxKernel);
    #endregion
    #region Matrix Kernels
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, int, int, float, float, int, int>> MatrixMultiplyKernel { get; private set; } = new(MatrixKernels.MatrixMultiplyKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int, float, float, int>> MatrixVectorMultiplyKernel { get; private set; } = new(MatrixKernels.MatrixVectorMultiplyKernel);

    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int>> TransposeKernel { get; private set; } =
        new(MatrixKernels.TransposeKernel);
    #endregion
    #region Vector Kernels
    public static KernelStorage<Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>, int, int>> OuterProductKernel { get; private set; } = new(VectorKernels.OuterProductKernel);
    #endregion
}