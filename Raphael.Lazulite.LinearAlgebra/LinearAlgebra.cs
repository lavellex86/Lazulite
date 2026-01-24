using System.Runtime.CompilerServices;

namespace Raphael.Lazulite.Suite;

public static partial class LinearAlgebra
{
    static LinearAlgebra()
    {
        InitializeCuBlas();
        AppDomain.CurrentDomain.ProcessExit += (_, _) => CleanupCuBlas();
    }
}