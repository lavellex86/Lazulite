using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public partial class KernelProgramming
{
    public static int MatrixIndexOf(int row, int col, int cols) => row * cols + col;
    public static (int row, int col) MatrixFromIndex(int index, int cols) => (index / cols, index % cols);
    
    public static float MatrixGet(ArrayView1D<float, Stride1D.Dense> array, int row, int col, int cols) => array[MatrixIndexOf(row, col, cols)];
    public static void MatrixSet(ArrayView1D<float, Stride1D.Dense> array, int row, int col, int cols, float value) => array[MatrixIndexOf(row, col, cols)] = value;
}