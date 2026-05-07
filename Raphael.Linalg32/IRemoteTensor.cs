using Raphael.Lazulite;

namespace Raphael.Linalg32;

public interface IRemoteTensor : IRemoteBase<float>
{
    public int[] Shape { get; }
    
    public IRemoteTensor Create(FMB buffer, BufferPool<float> pool, int[] shape);
    public IRemoteTensor Create(int[] shape, BufferPool<float> pool);
    public IRemoteTensor Create(int[] shape);
}