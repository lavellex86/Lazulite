namespace Lavelle.Lazulite;

/// <summary>
/// Holds a loaded kernel under a context.
/// </summary>
/// <typeparam name="T">The action type.</typeparam>
/// <param name="action">The action.</param>
/// <param name="lctx">The context under which this kernel runs.</param>
public class LazuliteKernel<T>(T action, LazuliteContext lctx)
{
    internal LazuliteContext _lctx = lctx;
    internal T _action = action;
    internal T? _compiled;
}