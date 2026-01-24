using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public static partial class KernelProgramming
{
    public static (float x, float y, float z) Vector3Get(ArrayView1D<float, Stride1D.Dense> array, int i)
    {
        int start = i * 3;
        return (array[start], array[start + 1], array[start + 2]);
    }
    public static void Vector3Set(ArrayView1D<float, Stride1D.Dense> array, int i, (float x, float y, float z) value)
    {
        int start = i * 3;
        (array[start], array[start + 1], array[start + 2]) = value;
    }

    public static (float x, float y, float z) Vector3Add((float x, float y, float z) a, (float x, float y, float z) b) => (a.x + b.x, a.y + b.y, a.z + b.z);
    public static (float x, float y, float z) Vector3Subtract((float x, float y, float z) a, (float x, float y, float z) b) => (a.x - b.x, a.y - b.y, a.z - b.z);
    public static (float x, float y, float z) Vector3Multiply((float x, float y, float z) a, float b) => (a.x * b, a.y * b, a.z * b);
    public static (float x, float y, float z) Vector3Divide((float x, float y, float z) a, float b) => (a.x / b, a.y / b, a.z / b);
    public static float Vector3Magnitude2((float x, float y, float z) a) => a.x * a.x + a.y * a.y + a.z * a.z;
    public static (float x, float y, float z) Vector3Negate((float x, float y, float z) a) => (-a.x, -a.y, -a.z);
    
}