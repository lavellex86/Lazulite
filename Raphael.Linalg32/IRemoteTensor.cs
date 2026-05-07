using Raphael.Lazulite;

namespace Raphael.Linalg32;

public interface IRemoteTensor : IRemoteBase<float>
{
    public int[] Shape { get; }
}