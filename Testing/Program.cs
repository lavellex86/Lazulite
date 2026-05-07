using System;
using Raphael.Lazulite;
using Raphael.Linalg32;

namespace Testing;

public static class Program
{
    public static void Main()
    {
        // claude 4.5 in copilot wrote this test suite up- if you couldn't tell
        
        using var ctx = new LazuliteContext()
            .EnableLinalg32();
        var tests = new SimpleTests(ctx);

        Console.WriteLine("╔════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║         LAZULITE & LINALG32 COMPREHENSIVE TEST SUITE               ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝");

        // Quick verification
        Console.WriteLine("\n════════ QUICK VERIFICATION ════════");
        tests.ElementwiseTest1();

        // Arithmetic operations tests
        Console.WriteLine("\n════════ ARITHMETIC OPERATIONS ════════");
        tests.TestAddition();
        tests.TestSubtraction();
        tests.TestMultiplication();
        tests.TestDivision();
        tests.TestScalarOperations();

        // Mathematical functions tests
        Console.WriteLine("\n════════ MATHEMATICAL FUNCTIONS ════════");
        tests.TestTranscendentalFunctions();
        tests.TestHyperbolicFunctions();
        tests.TestRoundingFunctions();
        tests.TestPower();

        // Min/Max and vector operations
        Console.WriteLine("\n════════ MIN/MAX & VECTOR OPERATIONS ════════");
        tests.TestMinMax();
        tests.TestFill();
        tests.TestConcat();
        tests.TestSlice();
        tests.TestNegate();
        tests.TestOuterProduct();

        // Performance benchmarks
        Console.WriteLine("\n════════ PERFORMANCE BENCHMARKS ════════");
        tests.BenchmarkElementwiseOperations();
        tests.BenchmarkTranscendentalFunctions();
        tests.BenchmarkDifferentDataSizes();

        // Stress tests
        Console.WriteLine("\n════════ STRESS TESTS ════════");
        tests.StressTestMemoryPool();
        tests.StressTestLargeOperations();

        Console.WriteLine("\n╔════════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                      TEST SUITE COMPLETED                          ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════════╝");
    }
}