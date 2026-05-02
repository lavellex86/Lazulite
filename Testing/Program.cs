using Raphael.Lazulite;
using Raphael.Linalg32;

namespace Testing;

public static class Program
{
    public static void Main()
    {
        using var ctx = new LazuliteContext()
            .EnableLinalg32();
        var simpleTests = new SimpleTests(ctx);
        
        simpleTests.ElementwiseTest1();
        simpleTests.ElementwiseTest1();
    }
}