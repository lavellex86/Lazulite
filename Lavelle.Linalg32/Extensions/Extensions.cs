using ILGPU.Runtime.Cuda;
using Lavelle.Lazulite;

namespace Lavelle.Linalg32;

/// <summary>
/// A set of linear algebra extensions for the <c>Lavelle.Lazulite</c> library.
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
        try { if (!_cuBlasDict.ContainsKey(lctx)) _cuBlasDict[lctx] = new((CudaAccelerator)lctx.Accelerator); }
        catch { }
        return lctx;
    }

    /// <summary>
    /// Inteprets this tensor as a scalar.
    /// </summary>
    public static RemoteScalar AsScalar<T>(this RemoteTensor<T> tensor) where T : notnull
    {
        if (tensor.Length != 1) throw new ArgumentException("Tensor is not length one", nameof(tensor));
        else return new(tensor, tensor.Pool);
    }

    /// <summary>
    /// Interprets this tensor as a vector.
    /// </summary>
    public static RemoteVector AsVector<T>(this RemoteTensor<T> tensor) where T : notnull
    {
        if (tensor is RemoteVector vector) return vector;
        else return new(tensor, tensor.Pool);
    }
    /// <summary>
    /// Interprets this tensor as a matrix.
    /// </summary>
    public static RemoteMatrix AsMatrix<T>(this RemoteTensor<T> tensor) where T : notnull
    {
        if (tensor is RemoteMatrix matrix) return matrix;
        if (tensor is RemoteTensor<float[]> vector) return new(vector.Buffer, vector.Pool, vector.Length);
        else throw new ArgumentException("Tensor cannot be interpeted as a matrix", nameof(tensor));
    }
}