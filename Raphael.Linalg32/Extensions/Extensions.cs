using ILGPU.Runtime.Cuda;
using Raphael.Lazulite;

namespace Raphael.Linalg32;

/// <summary>
/// A set of linear algebra extensions for the <c>Raphael.Lazulite</c> library.
/// </summary>
public static partial class LinalgExtensions
{
    private static Dictionary<LazuliteContext, BufferPool<float>> _pools = [];
    private static Dictionary<LazuliteContext, Kernels> _kernels = [];
    private static Dictionary<LazuliteContext, CuBlas<CuBlasPointerModeHandlers.AutomaticMode>> _cuBlasDict = [];

    private static BufferPool<float> GetPool(this LazuliteContext lctx) => _pools[lctx];
    private static Kernels GetKernels(this LazuliteContext lctx) => _kernels[lctx];
    private static CuBlas<CuBlasPointerModeHandlers.AutomaticMode> GetCuBlas(this LazuliteContext lctx) => _cuBlasDict[lctx];

    /// <summary>
    /// Retrives a scalar from the context pool.
    /// </summary>
    
    public static RemoteScalar GetScalar(this LazuliteContext lctx, bool cleared = false) => 
        new(lctx.GetPool().Get(1, cleared), lctx.GetPool());

    /// <summary>
    /// Retrives a vector from the context pool.
    /// </summary>
    public static RemoteVector GetVector(this LazuliteContext lctx, int length, bool cleared = false) =>
        new(lctx.GetPool().Get(length, cleared), lctx.GetPool());

    /// <summary>
    /// Retrives a matrix from the context pool.
    /// </summary>
    public static RemoteMatrix GetMatrix(this LazuliteContext lctx, int m0, int m1, bool cleared = false) =>
        new(lctx.GetPool().Get(m0 * m1, cleared), lctx.GetPool(), m0);

    /// <summary>
    /// Enables the <c>Linalg32</c> library on this context.
    /// </summary>
    /// <param name="lctx"></param>
    /// <returns></returns>
    public static LazuliteContext EnableLinalg32(this LazuliteContext lctx)
    {
        if (!_pools.ContainsKey(lctx)) _pools[lctx] = new(lctx);
        if (!_kernels.ContainsKey(lctx)) _kernels[lctx] = new(lctx);
        return lctx;
    }

    /// <summary>
    /// Interprets this tensor as a vector.
    /// </summary>
    public static RemoteVector AsVector<T>(this RemoteTensor<T> tensor) where T : notnull => new(tensor.Buffer, tensor.Pool);
    /// <summary>
    /// Interprets this vector as a matrix.
    /// </summary>
    public static RemoteMatrix AsMatrix(this RemoteTensor<float[]> tensor)  => new(tensor.Buffer, tensor.Pool, tensor.Length);

    /// <summary>
    /// Casts a <see cref="RemoteTensor{T}"/> back to a <see cref="RemoteScalar"/>.
    /// </summary>
    public static RemoteScalar CastScalar(this RemoteTensor<float> scalar) => (RemoteScalar)scalar;
    /// <summary>
    /// Casts a <see cref="RemoteTensor{T}"/> back to a <see cref="RemoteVector"/>.
    /// </summary>
    public static RemoteVector CastVector(this RemoteTensor<float[]> vector) => (RemoteVector)vector;
    /// <summary>
    /// Casts a <see cref="RemoteTensor{T}"/> back to a <see cref="RemoteMatrix"/>.
    /// </summary>
    public static RemoteMatrix CastMatrix(this RemoteTensor<float[,]> matrix) => (RemoteMatrix)matrix;
}