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
        // Declare kernel for correlation calculations
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV, FAV, FAV, FAV, FAV>> _pearsonKernel;

        // pearson, spearman, correlation matrix

        public RemoteScalar Pearson(RemoteVector x_data, RemoteVector y_data)
        {
            using var x_mean = Mean(x_data);
            using var y_mean = Mean(y_data);

            using var r_nomin = LContext.GetVector(x_data.Length);
            using var r_denom_var_x = LContext.GetVector(x_data.Length);
            using var r_denom_var_y = LContext.GetVector(y_data.Length);

            _pearsonKernel.Call(x_data.Length, r_nomin, r_denom_var_x, r_denom_var_y, x_data, y_data, x_mean, y_mean);

            using var nomin_sum = LContext.Sum(r_nomin).AsScalar();
            using var denom_var_x_sum = LContext.Sum(r_denom_var_x).AsScalar();
            using var denom_var_y_sum = LContext.Sum(r_denom_var_y).AsScalar();
            using var denom_prod = LContext.Multiply(denom_var_x_sum, denom_var_y_sum).AsScalar();
            using var denom_sqrt = LContext.Sqrt(denom_prod).AsScalar();

            return LContext.Divide(nomin_sum, denom_sqrt).AsScalar();
        }
    }
}
