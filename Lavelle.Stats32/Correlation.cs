using ILGPU;
using ILGPU.Runtime;
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
        private LazuliteKernel<Action<Index1D, FAV, FAV, FAV>> _spearmanKernel;

        private RemoteVector Rank(RemoteVector data)
        {
            float[] h_data = data.Get();
            int n = h_data.Length;
            int[] indices = new int[n];
            float[] ranked = new float[n];

            for (int i = 0; i < n; i++)
            {
                indices[i] = i;
            }

            /* Sort indices based on the values in h_data */
            Array.Sort(indices, (a, b) => h_data[a].CompareTo(h_data[b]));

            for (int i = 0; i < n; i++)
            {
                ranked[indices[i]] = i + 1; // Rank starts from 1
            }

            var resultVector = (RemoteVector) LContext.GetVector(n).Set(ranked);
            LContext.Synchronize();
            return resultVector;
        }

        private RemoteVector GetMatrixRow(RemoteMatrix matrix, int rowIndex)
        {
            int numCols = matrix.Shape[1];
            int offset = rowIndex * numCols;
            var vec = LContext.GetVector(numCols, true);
            matrix.Buffer.View.SubView(offset, numCols).CopyTo(vec);
            return vec;
        }

        // pearson, spearman, correlation matrix
        public RemoteScalar Pearson(RemoteVector x_data, RemoteVector y_data)
        {
            using var x_mean = Mean(x_data);
            using var y_mean = Mean(y_data);

            using var r_numer = LContext.GetVector(x_data.Length);
            using var r_denom_var_x = LContext.GetVector(x_data.Length);
            using var r_denom_var_y = LContext.GetVector(y_data.Length);

            _pearsonKernel.Call(x_data.Length, r_numer, r_denom_var_x, r_denom_var_y, x_data, y_data, x_mean, y_mean);
            LContext.Synchronize();
            
            using var numer_sum = LContext.Sum(r_numer).AsScalar();
            using var denom_var_x_sum = LContext.Sum(r_denom_var_x).AsScalar();
            using var denom_var_y_sum = LContext.Sum(r_denom_var_y).AsScalar();

            using var denom_prod = LContext.GetScalar(true);
            LContext.Multiply(denom_var_x_sum, denom_var_y_sum, r: denom_prod);


            using var denom_sqrt = LContext.GetScalar(true);
            LContext.Sqrt(denom_prod, r: denom_sqrt);

            var result = LContext.GetScalar(true);
            LContext.Divide(numer_sum, denom_sqrt, r: result);

            LContext.Synchronize();
            return result;
        }

        public RemoteScalar Spearman(RemoteVector x_data, RemoteVector y_data)
        {
            using var x_rank = Rank(x_data);
            using var y_rank = Rank(y_data);
            using var r_diff_squared = LContext.GetVector(x_data.Length);
            
            _spearmanKernel.Call(x_data.Length, r_diff_squared, x_rank, y_rank);
            LContext.Synchronize();

            using var sum_diff_squared = LContext.GetScalar(true);
            LContext.Sum(r_diff_squared, r: sum_diff_squared);

            float n = x_data.Length;
            float r_denom = n * (n * n - 1f);

            using var r_numerator = LContext.GetScalar(true);
            LContext.MultiplyScalar(sum_diff_squared, -6f, r: r_numerator);

            using var divided_r = LContext.GetScalar(true);
            LContext.DivideScalar(r_numerator, r_denom, r: divided_r);

            var result = LContext.GetScalar(true);
            LContext.AddScalar(divided_r, 1f, r: result);

            LContext.Synchronize();
            return result;
        }

        public RemoteMatrix PearsonMatrix(RemoteMatrix matrix)
        {
            if (matrix.Shape[0] != matrix.Shape[1])
            {
                throw new ArgumentException("Input matrix must be square for correlation matrix calculation.");
            }

            var dataVectors = new RemoteVector[matrix.Shape[0]];

            for (int i = 0; i < matrix.Shape[0]; i++)
            {
                dataVectors[i] = GetMatrixRow(matrix, i);
            }
            int n = dataVectors.Length;
            float[,] tmp_matrix = new float[n, n];

            for (int i = 0; i < n; i++)
            {
               for (int j = i; j < n; j++)
                {
                    if (i == j)
                    {
                        tmp_matrix[i, j] = 1.0f;
                    }
                    else
                    {
                        using var correlation = Pearson(dataVectors[i], dataVectors[j]);
                        float corr_value = correlation.Get();
                        tmp_matrix[i, j] = corr_value;
                        tmp_matrix[j, i] = corr_value;
                    }
                }
            }
            var result = LContext.GetMatrix(n, n, true);
            result.Set(tmp_matrix);
            LContext.Synchronize();
            return result;
        }
        public RemoteMatrix SpearmanMatrix(RemoteMatrix matrix)
        {
            if (matrix.Shape[0] != matrix.Shape[1])
            {
                throw new ArgumentException("Input matrix must be square for correlation matrix calculation.");
            }

            var dataVectors = new RemoteVector[matrix.Shape[0]];

            for (int i = 0; i < matrix.Shape[0]; i++)
            {
                dataVectors[i] = GetMatrixRow(matrix, i);
            }
            int n = dataVectors.Length;
            float[,] tmp_matrix = new float[n, n];

            for (int i = 0; i < n; i++)
            {
               for (int j = i; j < n; j++)
                {
                    if (i == j)
                    {
                        tmp_matrix[i, j] = 1.0f;
                    }
                    else
                    {
                        using var correlation = Spearman(dataVectors[i], dataVectors[j]);
                        float corr_value = correlation.Get();
                        tmp_matrix[i, j] = corr_value;
                        tmp_matrix[j, i] = corr_value;
                    }
                }
            }
            var result = LContext.GetMatrix(n, n, true);
            result.Set(tmp_matrix);
            LContext.Synchronize();
            return result;
        }
    }
}
