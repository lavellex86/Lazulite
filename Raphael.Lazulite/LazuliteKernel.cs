namespace Raphael.Lazulite;

public class LazuliteKernel<T>(T action, LazuliteContext lctx)
{
    internal LazuliteContext _lctx = lctx;
    internal T _action = action;
    internal T? _compiled;
}