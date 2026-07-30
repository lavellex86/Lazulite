using ILGPU;
using Raphael.Lazulite;
using Raphael.Linalg32;
using System;
using System.Collections.Generic;
using System.Text;

namespace Raphael.Calc32
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
        public RemoteVector VerletStep(RemoteVector FPrev, RemoteVector FPrePrev, RemoteVector d2FPrev, float dx)
        {
            var F = Context.GetVector(FPrev.Length);
            _verletKernel.Call(F.Length, F, FPrev, FPrePrev, d2FPrev, dx);
            return F;
        }
        public (RemoteVector F, RemoteVector dF) VelVerletStep(RemoteVector FPrev, RemoteVector dFPrev, RemoteVector d2FPrev, RemoteVector d2F, float dx)
        {
            var F = Context.GetVector(FPrev.Length);
            var dF = Context.GetVector(FPrev.Length);
            _velVerletKernel.Call(F.Length, F, dF, FPrev, dFPrev, d2FPrev, d2F, dx);
            return (F, dF);
        }


        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float>> _eulerKernel = new((i, F, f, prev, dx) => F[i] = prev[i] + f[i] * dx, ctx);
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, float>> _verletKernel = new((i, F, FPrev, FPrePrev, d2FPrev, dx) => F[i] = 2 * FPrev[i] - FPrePrev[i] + d2FPrev[i] * dx * dx, ctx);
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, FAV, FAV, float>> _velVerletKernel = new((i, F, dF, FPrev, dFPrev, d2FPrev, d2F, dx) =>
        {
            F[i] = FPrev[i] + dFPrev[i] * dx + 0.5f * d2FPrev[i] * dx * dx;
            dF[i] = dFPrev[i] + 0.5f * (d2FPrev[i] + d2F[i]) * dx;
        }, ctx);
    }

    public enum IntegrationMethod
    {
        Euler, Verlet, VelVerlet,
        RK, RK2, RK3, RK4
    }
}
