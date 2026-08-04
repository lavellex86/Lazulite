using ILGPU;
using Lavelle.Lazulite;
using Lavelle.Linalg32;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Calc32
{
    public class IntegrationContext(LazuliteContext ctx)
    {
        public LazuliteContext Context { get; set; } = ctx;

        public RemoteVector EulerStep(RemoteVector f, RemoteVector prevF, float dx)
        {
            var F = Context.GetVector(f.Length);
            _eulerKernel.Call(F.Length, F, f, prevF, dx);
            return F;
        }
        public RemoteVector VerletStep(RemoteVector FPrev, RemoteVector FPrePrev, RemoteVector dfPrev, float dx)
        {
            var F = Context.GetVector(FPrev.Length);
            _verletKernel.Call(F.Length, F, FPrev, FPrePrev, dfPrev, dx);
            return F;
        }
        public (RemoteVector F, RemoteVector dF) VelVerletStep(RemoteVector FPrev, RemoteVector fPrev, RemoteVector dfPrev, RemoteVector df, float dx)
        {
            var F = Context.GetVector(FPrev.Length);
            var dF = Context.GetVector(FPrev.Length);
            _velVerletKernel.Call(F.Length, F, dF, FPrev, fPrev, dfPrev, df, dx);
            return (F, dF);
        }

        public RemoteVector[] EulerIntegrate(RemoteVector[] f, RemoteVector initialF, float dx)
        {
            var steps = f.Length;
            var F = new RemoteVector[steps];
            F[0] = initialF;
            for (int i = 1; i < steps; i++) F[i] = EulerStep(f[i], F[i], dx);
            return F;
        }

        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float>> _eulerKernel = new((i, F, f, prev, dx) => F[i] = prev[i] + f[i] * dx, ctx);
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, float>> _verletKernel = new((i, F, FPrev, FPrePrev, dfPrev, dx) => F[i] = 2 * FPrev[i] - FPrePrev[i] + dfPrev[i] * dx * dx, ctx);
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, FAV, FAV, float>> _velVerletKernel = new((i, F, f, FPrev, fPrev, dfPrev, df, dx) =>
        {
            F[i] = FPrev[i] + fPrev[i] * dx + 0.5f * dfPrev[i] * dx * dx;
            f[i] = fPrev[i] + 0.5f * (dfPrev[i] + df[i]) * dx;
        }, ctx);
    }

    public enum IntegrationMethod
    {
        Euler, Verlet, VelVerlet,
        RK, RK2, RK3, RK4
    }
}
