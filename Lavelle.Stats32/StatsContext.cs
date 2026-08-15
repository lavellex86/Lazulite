using ILGPU.Algorithms;
using Lavelle.Lazulite;
using System;
using System.Collections.Generic;
using System.Text;

namespace Lavelle.Stats32
{
    public partial class StatsContext
    {
        public LazuliteContext LContext { get; }

        public StatsContext(LazuliteContext lctx)
        {
            LContext = lctx;

            _varianceKernel = new((i, r, data, mean, denom) => r[i] = (data[i] - mean[0]) * (data[i] - mean[0]) / denom, lctx);
            _pcgKernel = new((i, r, seed) => r[i] = Pcg(seed, (ulong)i), lctx);
            _normalKernel = new((i, r, seed, sigma, mu) =>
            {
                float u1 = Pcg(seed, 2 * (ulong)i) , u2 = Pcg(seed, 2 * (ulong)i + 1);
                var mag = XMath.Sqrt(-2f * XMath.Log(u1));
                r[2 * i] = mag * XMath.Cos(2f * XMath.PI * u2) * sigma + mu;
                if (2 * i + 1 < r.Length) r[2 * i + 1] = mag * XMath.Sin(2f * XMath.PI * u2) * sigma + mu;
            }, lctx);
            _bernoulliKernel = new((i, r, u, p) => r[i] = u[i] < p ? 1f : 0f, lctx);
            _pearsonKernel = new((i, nomin, denom_var_x, denom_var_y, x_vec, y_vec, x_mean, y_mean) =>
            {
                float dx = x_vec[i] - x_mean[0];
                float dy = y_vec[i] - y_mean[0];

                nomin[i] = dx * dy;
                denom_var_x[i] = dx * dx;
                denom_var_y[i] = dy * dy;

            }, lctx);
            _spearmanKernel = new((i, result, x_ranked, y_ranked) =>
            {
                result[i] = (x_ranked[i] - y_ranked[i]) * (x_ranked[i] - y_ranked[i]);
            }, lctx);
        }

        private static float Pcg(ulong seed, ulong i)
        {
            var state = seed + i;
            const ulong mul = 6364136223846793005UL;
            const ulong inc = 1442695040888963407UL;

            state = state * mul + inc;
            state = state * mul + inc;
            var xorshift = (uint)(((state >> 18) ^ state) >> 27);
            var rotation = (int)(state >> 59);
            var random = (xorshift >> rotation) | (xorshift << (-rotation & 31));
            return (random >> 8) / 16777216f;
        }
    }
}
