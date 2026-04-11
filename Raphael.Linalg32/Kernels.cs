
using ILGPU;
using Raphael.Lazulite;

namespace Raphael.Linalg32;

internal class Kernels(LazuliteContext lctx)
{
    internal LazuliteKernel<Action<Index1D, FAV, float>> _fillKernel = new(Implementations.Fill, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _concatKernel = new(Implementations.Concat, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, int, int>> _sliceKernel = new(Implementations.Slice, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _addKernel = new(Implementations.Add, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _subtractKernel = new(Implementations.Subtract, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _multiplyKernel = new(Implementations.Multiply, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _divideKernel = new(Implementations.Divide, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _addScalarKernel = new(Implementations.AddScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _subtractScalarKernel = new(Implementations.SubtractScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _multiplyScalarKernel = new(Implementations.MultiplyScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _divideScalarKernel = new(Implementations.DivideScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _expKernel = new(Implementations.Exp, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _logKernel = new(Implementations.Log, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _log10Kernel = new(Implementations.Log10, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _log2Kernel = new(Implementations.Log2, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _sqrtKernel = new(Implementations.Sqrt, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _powKernel = new(Implementations.Pow, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _powScalarKernel = new(Implementations.PowScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _sinKernel = new(Implementations.Sin, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _cosKernel = new(Implementations.Cos, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _tanKernel = new(Implementations.Tan, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _sinhKernel = new(Implementations.Sinh, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _coshKernel = new(Implementations.Cosh, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _tanhKernel = new(Implementations.Tanh, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _absKernel = new(Implementations.Abs, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _floorKernel = new(Implementations.Floor, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _ceilingKernel = new(Implementations.Ceiling, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _roundKernel = new(Implementations.Round, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _truncateKernel = new(Implementations.Truncate, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _signKernel = new(Implementations.Sign, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _minKernel = new(Implementations.Min, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _maxKernel = new(Implementations.Max, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _minScalarKernel = new(Implementations.MinScalar, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _maxScalarKernel = new(Implementations.MaxScalar, lctx);

    internal LazuliteKernel<Action<Index1D, FAV, FAV>> _negateKernel = new(Implementations.Negate, lctx);
    
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV, int>> _outerProductKernel = new(Implementations.OuterProduct, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, float>> _axpyKernel = new(Implementations.Axpy, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, FAV, int, int, float, float, int>> _matrixMultiplyKernel = new(Implementations.MatrixMultiply, lctx);
    internal LazuliteKernel<Action<Index1D, FAV, FAV, int>> _transposeKernel = new(Implementations.Transpose, lctx);
}