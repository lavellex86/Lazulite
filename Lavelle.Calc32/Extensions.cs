using Lavelle.Lazulite;

namespace Lavelle.Calc32;

public static class CalcExtensions
{
    private static Dictionary<LazuliteContext, IntegrationContext> _ints = [];

    public static LazuliteContext EnableCalc32(this LazuliteContext lctx)
    {
        if (!_ints.ContainsKey(lctx)) _ints[lctx] = new(lctx);
        return lctx;
    }

    public static IntegrationContext GetIntegrationContext(this LazuliteContext lctx) => _ints[lctx];
}
