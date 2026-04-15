
using ILGPU;
using Raphael.Lazulite;

namespace Raphael.Linalg32;

internal class Kernels(LazuliteContext lctx)
{
    internal LazuliteKernel<Action<Index1D, FAV, float>> FillKernel = new(Implementations.Fill, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> ConcatKernel = new(Implementations.Concat, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, int, int>> SliceKernel = new(Implementations.Slice, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> AddKernel = new(Implementations.Add, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> SubtractKernel = new(Implementations.Subtract, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MultiplyKernel = new(Implementations.Multiply, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> DivideKernel = new(Implementations.Divide, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> AddScalarKernel = new(Implementations.AddScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> SubtractScalarKernel = new(Implementations.SubtractScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> MultiplyScalarKernel = new(Implementations.MultiplyScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> DivideScalarKernel = new(Implementations.DivideScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> ExpKernel = new(Implementations.Exp, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> LogKernel = new(Implementations.Log, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> Log10Kernel = new(Implementations.Log10, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> Log2Kernel = new(Implementations.Log2, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> SqrtKernel = new(Implementations.Sqrt, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> PowKernel = new(Implementations.Pow, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> PowScalarKernel = new(Implementations.PowScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> SinKernel = new(Implementations.Sin, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> CosKernel = new(Implementations.Cos, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> TanKernel = new(Implementations.Tan, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> SinhKernel = new(Implementations.Sinh, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> CoshKernel = new(Implementations.Cosh, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> TanhKernel = new(Implementations.Tanh, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> AbsKernel = new(Implementations.Abs, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> FloorKernel = new(Implementations.Floor, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> CeilingKernel = new(Implementations.Ceiling, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> RoundKernel = new(Implementations.Round, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> TruncateKernel = new(Implementations.Truncate, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> SignKernel = new(Implementations.Sign, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MinKernel = new(Implementations.Min, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MaxKernel = new(Implementations.Max, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> MinScalarKernel = new(Implementations.MinScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> MaxScalarKernel = new(Implementations.MaxScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> NegateKernel = new(Implementations.Negate, lctx);
    
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV, int>> OuterProductKernel = new(Implementations.OuterProduct, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> AxpyKernel = new(Implementations.Axpy, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV, int, int, float, float, int>> MatrixMultiplyKernel = new(Implementations.MatrixMultiply, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, int>> TransposeKernel = new(Implementations.Transpose, lctx);
}