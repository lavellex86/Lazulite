using Lavelle.Calc32;
using Lavelle.Lazulite;
using Lavelle.Linalg32;

var lctx = new LazuliteContext();
var cctx = new CalcContext(lctx);

var f1 = new RemoteVector[]
{
    lctx.GetVector(1).Set([1]).AsVector(),
    lctx.GetVector(1).Set([2]).AsVector(),
    lctx.GetVector(1).Set([3]).AsVector(),
    lctx.GetVector(1).Set([4]).AsVector()
};
var df1 = cctx.Differentiate(f1, 0.01f);

var f2 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 1, 1]).AsVector(),
    lctx.GetVector(3).Set([2, 2, 2]).AsVector(),
    lctx.GetVector(3).Set([1, 1, 1]).AsVector()
};
var initialF2 = lctx.GetVector(3).Set([0, 0, 0]).AsVector();
var F2 = cctx.EulerIntegrate(f2, initialF2, 0.01f);