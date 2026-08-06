using ILGPU;
using Lavelle.Lazulite;
using Lavelle.Linalg32;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text;

namespace Lavelle.Calc32
{
    public partial class CalcContext
    {
        #region Steps
        /// <summary>
        /// Takes an Euler integration step using the current function value <paramref name="f"/> and previous integral value <paramref name="prevF"/>.
        /// </summary>
        public RemoteVector EulerStep(RemoteVector f, RemoteVector prevF, float dx)
        {
            var F = LContext.GetVector(f.Length);
            _eulerKernel.Call(F.Length, F, f, prevF, dx);
            return F;
        }

        /// <summary>
        /// Takes a Verlet integration step using the previous integral value <paramref name="prevF"/>, 
        /// pre-previous integral value <paramref name="prePrevF"/>, and the previous first-order derivative value <paramref name="prevdf"/>.
        /// </summary>
        public RemoteVector VerletStep(RemoteVector prevF, RemoteVector prePrevF, RemoteVector prevdf, float dx)
        {
            var F = LContext.GetVector(prevF.Length);
            _verletKernel.Call(F.Length, F, prevF, prePrevF, prevdf, dx);
            return F;
        }
        /// <summary>
        /// Takes a velocity Verlet step using the previous integral value <paramref name="prevF"/>, previous function value <paramref name="prevf"/>, 
        /// and previous first-order derivative value <paramref name="prevdf"/>.
        /// </summary>
        public (RemoteVector F, RemoteVector f) VelVerletStep(RemoteVector prevF, RemoteVector prevf, RemoteVector prevdf, RemoteVector df, float dx)
        {
            var F = LContext.GetVector(prevF.Length);
            var f = LContext.GetVector(prevF.Length);
            _velVerletKernel.Call(F.Length, F, f, prevF, prevf, prevdf, df, dx);
            return (F, f);
        }
        #endregion
        #region Integrations
        /// <summary>
        /// Takes the integral of <paramref name="f"/> with Euler's method.
        /// </summary>
        public RemoteVector[] EulerIntegrate(RemoteVector[] f, RemoteVector initialF, float dx)
        {
            var steps = f.Length;
            var F = new RemoteVector[steps];
            F[0] = initialF;
            for (int i = 1; i < steps; i++) F[i] = EulerStep(f[i], F[i - 1], dx);
            return F;
        }

        /// <summary>
        /// Takes the integral of a function using it's first-order derivative <paramref name="df"/> using Verlet's method. 
        /// </summary>
        public RemoteVector[] VerletIntegrate(RemoteVector[] df, RemoteVector initialF, RemoteVector initialf, float dx)
        {
            var steps = df.Length;
            var F = new RemoteVector[steps];
            F[0] = initialF;
            F[1] = EulerStep(initialf, initialF, dx);
            for (int i = 2; i < steps; i++) F[i] = VerletStep(F[i - 1], F[i - 2], df[i - 1], dx);
            return F;
        }

        /// <summary>
        /// Takes the integral of a function using it's first-order derivative <paramref name="df"/> using the Velocity Verlet method, returning both the integral and the function.
        /// </summary>
        public (RemoteVector[] F, RemoteVector[] f) VelVerletIntegrate(RemoteVector[] df, RemoteVector initialF, RemoteVector initialf, float dx)
        {
            var steps = df.Length;
            RemoteVector[] f = new RemoteVector[steps], F = new RemoteVector[steps];
            f[0] = initialf; F[0] = initialF;
            f[1] = EulerStep(df[1], f[0], dx); F[0] = EulerStep(f[1], F[0], dx);
            for (int i = 2; i < steps; i++) (F[i], f[i]) = VelVerletStep(F[i - 1], f[i - 1], df[i - 1], df[i], dx);
            return (F, f);
        }
        #endregion

        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float>> _eulerKernel;
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, float>> _verletKernel;
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, FAV, FAV, float>> _velVerletKernel;
        // RK2-4 using callbacks soon
    }
}
