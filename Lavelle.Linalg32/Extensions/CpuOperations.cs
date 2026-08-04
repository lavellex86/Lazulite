using Lavelle.Lazulite;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Lavelle.Linalg32
{
    public static partial class LinalgExtensions
    {
        /// <summary>
        /// Inverts a square matrix on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static RemoteTensor<float[,]> CpuInvert(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
        {
            var hostMatrix = matrix.Get();
            int n = hostMatrix.GetLength(0);
            var (lu, piv, _) = DecomposeLU(hostMatrix);

            var inv = new float[n, n];
            for (int col = 0; col < n; col++)
            {
                var x = new float[n];
                x[col] = 1f;

                for (int k = 0; k < n; k++) (x[k], x[piv[k]]) = (x[piv[k]], x[k]);

                for (int i = 1; i < n; i++)
                    for (int j = 0; j < i; j++) x[i] -= lu[i, j] * x[j];

                for (int i = n - 1; i >= 0; i--)
                {
                    for (int j = i + 1; j < n; j++) x[i] -= lu[i, j] * x[j];
                    x[i] /= lu[i, i];
                }

                for (int i = 0; i < n; i++) inv[i, col] = x[i];
            }

            return (RemoteMatrix)lctx.GetMatrix(n, n).Set(inv);
        }

        /// <summary>
        /// Computes the determinant of a square matrix on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static float CpuDet(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
        {
            var (lu, _, sign) = DecomposeLU(matrix.Get());
            int n = lu.GetLength(0);

            var det = (float)sign;
            for (int i = 0; i < n; i++) det *= lu[i, i];
            return det;
        }

        /// <summary>
        /// Takes the L1 norm of a vector (sum of absolutes) on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static float CpuL1Norm(this LazuliteContext lctx, RemoteTensor<float[]> vector)
        {
            var v = vector.Get();
            var sum = 0f;
            for (int i = 0; i < v.Length; i++) sum += MathF.Abs(v[i]);
            return sum;
        }

        /// <summary>
        /// Takes the L2 norm of a vector (sum of squares) on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static float CpuL2Norm(this LazuliteContext lctx, RemoteTensor<float[]> vector)
        {
            var v = vector.Get();
            var sum = 0f;
            for (int i = 0; i < v.Length; i++) sum += v[i] * v[i];
            return MathF.Sqrt(sum);
        }

        /// <summary>
        /// Takes the sum of a vector on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static float CpuSum(this LazuliteContext lctx, RemoteTensor<float[]> vector)
        {
            var v = vector.Get();
            float sum = 0f;
            for (int i = 0; i < v.Length; i++)
                sum += v[i];
            return sum;
        }

        /// <summary>
        /// Takes the trace of a matrix on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static float CpuTrace(this LazuliteContext lctx, RemoteTensor<float[,]> matrix)
        {
            var m = matrix.Get();
            var n = Math.Min(m.GetLength(0), m.GetLength(1));
            var sum = 0f;
            for (int i = 0; i < n; i++) sum += m[i, i];
            return sum;
        }

        /// <summary>
        /// Finds the index of the minimum element on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static int CpuArgMin(this LazuliteContext lctx, RemoteTensor<float[]> vector)
        {
            var v = vector.Get();
            var (minIdx, minVal) = (0, v[0]);
            for (int i = 1; i < v.Length; i++)
                if (v[i] < minVal) (minIdx, minVal) = (i, v[i]);
            return minIdx;
        }

        /// <summary>
        /// Finds the index of the maximum element on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static int CpuArgMax(this LazuliteContext lctx, RemoteTensor<float[]> vector)
        {
            var v = vector.Get();
            var (maxIdx, maxVal) = (0, v[0]);
            for (int i = 1; i < v.Length; i++)
                if (v[i] > maxVal) (maxIdx, maxVal) = (i, v[i]);
            return maxIdx;
        }

        /// <summary>
        /// Takes the LU decomposition of a matrix on the CPU (syncing and transferring it from the compute device).
        /// </summary>
        public static (RemoteMatrix lu, RemoteVector piv, int sign) CpuDecomposeLU(this LazuliteContext lctx, RemoteMatrix x)
        {
            (var lu, var piv, var sign) = DecomposeLU(x.Get());
            return (
                lctx.GetMatrix(lu.GetLength(0), lu.GetLength(0)).Set(lu).AsMatrix(),
                lctx.GetVector(piv.Length).Set([.. piv.Select(i => (float)i)]).AsVector(),
                sign);
        }

        private static (float[,] lu, int[] piv, int sign) DecomposeLU(float[,] x)
        {
            int n = x.GetLength(0);
            if (x.GetLength(1) != n) throw new ArgumentException("Matrix must be square.");

            var lu = (float[,])x.Clone();
            var piv = new int[n];
            int sign = 1;

            for (int k = 0; k < n; k++)
            {
                int p = k;
                for (int i = k + 1; i < n; i++)
                    if (MathF.Abs(lu[i, k]) > MathF.Abs(lu[p, k])) p = i;

                if (p != k)
                {
                    for (int j = 0; j < n; j++) (lu[k, j], lu[p, j]) = (lu[p, j], lu[k, j]);
                    sign = -sign;
                }
                piv[k] = p;

                if (lu[k, k] == 0f) throw new InvalidOperationException("Matrix is singular.");
                for (int i = k + 1; i < n; i++)
                {
                    lu[i, k] /= lu[k, k];
                    for (int j = k + 1; j < n; j++) lu[i, j] -= lu[i, k] * lu[k, j];
                }
            }

            return (lu, piv, sign);
        }

    }
}
