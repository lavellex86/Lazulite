
using ILGPU;
using Lavelle.Lazulite;
using static Lavelle.Linalg32.Implementations;

namespace Lavelle.Linalg32;

internal class Kernels(LazuliteContext lctx)
{
    internal readonly LazuliteKernel<Action<Index1D, FAV, float>> FillKernel = new(Fill, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> ConcatKernel = new(Concat, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, int>> SliceKernel = new(Slice, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> AddKernel = new(Add, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> SubtractKernel = new(Subtract, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MultiplyKernel = new(Multiply, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> DivideKernel = new(Divide, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> AddScalarKernel = new(AddScalar, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> SubtractScalarKernel = new(SubtractScalar, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> MultiplyScalarKernel = new(MultiplyScalar, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> DivideScalarKernel = new(DivideScalar, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> ExpKernel = new(Exp, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> LogKernel = new(Log, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> Log10Kernel = new(Log10, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> Log2Kernel = new(Log2, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> SqrtKernel = new(Sqrt, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> PowKernel = new(Pow, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> PowScalarKernel = new(PowScalar, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> SinKernel = new(Sin, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> CosKernel = new(Cos, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> TanKernel = new(Tan, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> SinhKernel = new(Sinh, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> CoshKernel = new(Cosh, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> TanhKernel = new(Tanh, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> AbsKernel = new(Abs, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> FloorKernel = new(Floor, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> CeilingKernel = new(Ceiling, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> RoundKernel = new(Round, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> TruncateKernel = new(Truncate, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> SignKernel = new(Sign, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MinKernel = new(Min, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> MaxKernel = new(Max, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> MinScalarKernel = new(MinScalar, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float>> MaxScalarKernel = new(MaxScalar, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> NegateKernel = new(Negate, lctx);
    
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float, float>> OuterProductKernel = new(OuterProduct, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV, int, int, float, float, int>> MatrixMultiplyKernel = new(MatrixMultiply, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, int>> TransposeKernel = new(Transpose, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> BroadcastMatrixVectorAddKernel = new(BroadcastMatrixVectorAdd, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, int>> NarrowcastVectorMatrixAddKernel = new(NarrowcastVectorMatrixAdd, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> DotKernel = new(Dot, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float>> AxpyKernel = new(Axpy, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> L1NormKernel = new(L1Norm, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> L2NormKernel = new(L2Norm, lctx);

    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, float, float>> ClampKernel = new(Clamp, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> ReciprocalKernel = new(Reciprocal, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> Atan2Kernel = new(Atan2, lctx);
    internal readonly LazuliteKernel<Action<Index1D, FAV, FAV>> SumKernel = new(Sum, lctx);
}