using ILGPU;
using Lavelle.Lazulite;
using Lavelle.Linalg32;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Calc32
{
    public partial class CalcContext
    {
        #region Steps
        /// <summary>
        /// Takes the forward difference of a function using a next and current value.
        /// </summary>
        public RemoteVector ForwardDifferenceStep(RemoteVector fNext, RemoteVector f, float dx)
        {
            var r = LContext.Subtract(fNext, f);
            return LContext.DivideScalar(r, dx, r).AsVector();
        }

        /// <summary>
        /// Takes the backward difference of a function using a previous and current value.
        /// </summary>
        public RemoteVector BackwardDifferenceStep(RemoteVector fPrev, RemoteVector f, float dx)
        {
            var r = LContext.Subtract(f, fPrev);
            return LContext.DivideScalar(r, dx, r).AsVector();
        }

        /// <summary>
        /// Takes the central difference of a function using a next and previous value.
        /// </summary>
        public RemoteVector CentralDifferenceStep(RemoteVector fNext, RemoteVector fPrev, float dx)
        {
            var r = LContext.Subtract(fNext, fPrev);
            return LContext.DivideScalar(r, 2f * dx, r).AsVector();
        }
        #endregion

        /// <summary>
        /// Takes the derivative of a function.
        /// </summary>
        public RemoteVector[] Differentiate(RemoteVector[] f, float dx)
        {
            var steps = f.Length;
            var df = new RemoteVector[steps];
            if (steps == 1) throw new ArgumentException($"Length-1 function cannot be differentiated");

            df[0] = ForwardDifferenceStep(f[1], f[0], dx);
            for (int i = 1; i < steps - 1; i++) df[i] = CentralDifferenceStep(f[i + 1], f[i - 1], dx);
            df[steps - 1] = BackwardDifferenceStep(f[steps - 2], f[steps - 1], dx);

            return df;
        }
    }
}
