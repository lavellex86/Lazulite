using ILGPU;
using Lavelle.Lazulite;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Calc32
{
    public partial class CalcContext
    {
        public LazuliteContext LContext { get; set; } 

        public CalcContext(LazuliteContext lctx)
        {
            LContext = lctx;

            _eulerKernel = new((i, F, f, prev, dx) => F[i] = prev[i] + f[i] * dx, lctx);
            _verletKernel = new((i, F, FPrev, FPrePrev, dfPrev, dx) => F[i] = 2 * FPrev[i] - FPrePrev[i] + dfPrev[i] * dx * dx, lctx);
            _velVerletKernel = new((i, F, f, FPrev, fPrev, dfPrev, df, dx) =>
            {
                F[i] = FPrev[i] + fPrev[i] * dx + 0.5f * dfPrev[i] * dx * dx;
                f[i] = fPrev[i] + 0.5f * (dfPrev[i] + df[i]) * dx;
            }, lctx);
    }
    }
}
