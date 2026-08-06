using ILGPU;
using Lavelle.Lazulite;
using Lavelle.Linalg32;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Stats32
{
    public partial class StatsContext
    {
        // bernoulli, binomial, poisson, beta, gamma, chi-squared next

        public RemoteVector UniformDistribution(int n, ulong? seed = null, RemoteVector? r = null)
        {
            r ??= LContext.GetVector(n);
            _pcgKernel.Call(n, r, seed ?? (ulong)Random.Shared.NextInt64());
            return r;
        }
        public RemoteVector ExponentialDistribution(int n, float lambda, ulong? seed = null, RemoteVector? r = null)
        {
            r ??= LContext.GetVector(n);
            using var u = UniformDistribution(n, seed);
            LContext.Log(r, r);
            LContext.Negate(r, r);
            LContext.DivideScalar(r, lambda, r);
            return r;
        }

        public RemoteVector NormalDistribution(int n, float sigma = 1f, float mu = 0f, ulong? seed = null, RemoteVector? r = null)
        {
            r ??= LContext.GetVector(n);
            _normalKernel.Call(n, r, seed ?? (ulong)Random.Shared.NextInt64(), sigma, mu);
            return r;
        }

        private LazuliteKernel<Action<Index1D, FAV, ulong>> _pcgKernel;
        private LazuliteKernel<Action<Index1D, FAV, ulong, float, float>> _normalKernel;
    }
}
