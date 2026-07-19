using ILGPU;
using Raphael.Lazulite;
using Raphael.Linalg32;
using System;
using System.Collections.Generic;
using System.Text;

namespace Raphael.Calc32
{
    public class Integration(LazuliteContext ctx)
    {
        public LazuliteContext Context { get; set; } = ctx;

        public RemoteVector EulerStep(RemoteVector f, RemoteScalar dx, RemoteVector prevF)
        {
            var F = Context.GetVector(f.IntLength);
            _eulerKernel.Call(F.IntLength, F, f, dx, prevF);
            return F;
        }

        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV>> _eulerKernel = new((i, F, f, dx, prev) => F[i] = prev[i] + f[i] * dx[0], ctx);
    }

    public enum IntegrationMethod
    {
        Euler, Verlet,
        RK, RK2, RK3, RK4
    }
}
