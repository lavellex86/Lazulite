using ILGPU.Runtime.Cuda;
using Raphael.Lazulite;

namespace Raphael.Linalg32;

public static partial class Extensions
{
    private static Dictionary<int, BufferPool<float>> _pools = [];
    private static Dictionary<int, Kernels> _kernels = [];
    private static Dictionary<int, CuBlas<CuBlasPointerModeHandlers.AutomaticMode>> _cuBlasDict = [];

    private static BufferPool<float> GetPool(this LazuliteContext lctx) => _pools[lctx.GetHashCode()];
    private static Kernels GetKernels(this LazuliteContext lctx) => _kernels[lctx.GetHashCode()];
    private static CuBlas<CuBlasPointerModeHandlers.AutomaticMode> GetCuBlas(this LazuliteContext lctx) => _cuBlasDict[lctx.GetHashCode()];
    
    public static RemoteScalar GetScalar(this LazuliteContext lctx, bool cleared = false) => 
        new(lctx.GetPool().Get(1, cleared), lctx.GetPool());

    public static RemoteVector GetVector(this LazuliteContext lctx, int length, bool cleared = false) =>
        new(lctx.GetPool().Get(length, cleared), lctx.GetPool());
    
    public static RemoteMatrix GetMatrix(this LazuliteContext lctx, int m0, int m1, bool cleared = false) =>
        new(lctx.GetPool().Get(m0 * m1, cleared), lctx.GetPool(), m0);

    public static LazuliteContext EnableLinalg32(this LazuliteContext lctx)
    {
        if (!_pools.ContainsKey(lctx.GetHashCode())) _pools[lctx.GetHashCode()] = new(lctx);
        if (!_kernels.ContainsKey(lctx.GetHashCode())) _kernels[lctx.GetHashCode()] = new(lctx);
        return lctx;
    }

    public static RemoteVector AsVector<T>(this RemoteTensor<T> tensor) where T : notnull => new(tensor.Buffer, tensor.Pool);
    public static RemoteMatrix AsMatrix<T>(this RemoteTensor<T> tensor) where T : notnull => new(tensor.Buffer, tensor.Pool, tensor.IntLength);

    public static RemoteScalar CastScalar(this RemoteTensor<float> scalar) => (RemoteScalar)scalar;
    public static RemoteVector CastVector(this RemoteTensor<float[]> scalar) => (RemoteVector)scalar;
    public static RemoteMatrix CastMatrix(this RemoteTensor<float[,]> scalar) => (RemoteMatrix)scalar;
}