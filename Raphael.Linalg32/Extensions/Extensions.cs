using Raphael.Lazulite;

namespace Raphael.Linalg32;

public static partial class Extensions
{
    private static Dictionary<int, BufferPool<float>> _pools;
    private static Dictionary<int, Kernels> _kernels;

    private static BufferPool<float> GetPool(this LazuliteContext lctx) => _pools[lctx.GetHashCode()];
    private static Kernels GetKernels(this LazuliteContext lctx) => _kernels[lctx.GetHashCode()];
    
    public static RemoteScalar GetScalar(this LazuliteContext lctx, bool cleared = false) => 
        new(lctx.GetPool().Get(1, cleared), lctx.GetPool());

    public static RemoteVector GetVector(this LazuliteContext lctx, int length, bool cleared = false) =>
        new(lctx.GetPool().Get(length, cleared), lctx.GetPool());
    
    public static RemoteMatrix GetMatrix(this LazuliteContext lctx, int m0, int m1, bool cleared = false) =>
        new(lctx.GetPool().Get(m0 * m1, cleared), lctx.GetPool(), m0);

    public static LazuliteContext EnableLinalg32(this LazuliteContext lctx)
    {
        _pools[lctx.GetHashCode()] = new(lctx);
        _kernels[lctx.GetHashCode()] = new(lctx);
        return lctx;
    }
}