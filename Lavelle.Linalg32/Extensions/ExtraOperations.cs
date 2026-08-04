using Lavelle.Lazulite;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Linalg32
{
    public partial class LinalgExtensions
    {
        /// <summary>
        /// Solves ||Ax - b||^2 for x.
        /// </summary>
        public static RemoteVector LeastSquares(this LazuliteContext lctx, RemoteMatrix a, RemoteVector b)
        {
            var aTa = lctx.MatrixMultiply(a, a, transposeA: true);
            var inv = lctx.CpuInvert(aTa);
            var psinv = lctx.MatrixMultiply(inv, a, transposeB: true);
            return lctx.MatrixVectorMultiply(psinv, b).AsVector();
        }
    }
}
