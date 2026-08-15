using ILGPU;
using Lavelle.Lazulite;
using Lavelle.Linalg32;
using System;
using System.Collections.Generic;
using System.Data;
using System.Text;

namespace Lavelle.Stats32
{
    public partial class StatsContext
    {
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, float>> _varianceKernel;

        public RemoteScalar Mean(RemoteVector data)
        {
            using var normalized = LContext.DivideScalar(data, data.Length);
            var sum = LContext.Sum(normalized).AsScalar();
            LContext.Synchronize();
            return sum;
        }

        public RemoteScalar Variance(RemoteVector data, bool sample = false)
        {
            using var mean = Mean(data);
            using var vec = LContext.GetVector(data.Length);
            _varianceKernel.Call(data.Length, vec, data, mean, sample ? data.Length - 1 : data.Length);
            return LContext.Sum(vec).AsScalar();
        }

        public RemoteScalar Stddev(RemoteVector data)
        {
            using var variance = Variance(data);
            return LContext.Sqrt(variance).AsScalar();
        }

        // skewness and kurtosis on gpu
    }
}
