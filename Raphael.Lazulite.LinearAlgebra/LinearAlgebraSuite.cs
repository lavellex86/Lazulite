using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite.LinearAlgebra;

public static partial class LinearAlgebraSuite
{
    static LinearAlgebraSuite()
    {
        InitializeCuBlas();
        AppDomain.CurrentDomain.ProcessExit += (_, _) => CleanupCuBlas();
    }
}