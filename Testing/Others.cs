using static Raphael.Lazulite.KernelProgramming;

namespace Testing;

public class Others
{
    public static void IndexingTest()
    {
        Console.WriteLine("Indexing Test 1: Strided/Interleaved (x0,y0,x1,y1,...)");
        int[] interleaved = new int[8];
        for (int i = 0; i < 4; i++)
        {
            interleaved[StridedIndexOf(i, 2, 0)] = i;
            interleaved[StridedIndexOf(i, 2, 1)] = i + 10;
        }
        Console.WriteLine($"Array: [{string.Join(", ", interleaved)}]");
        Console.WriteLine($"x2 at index {StridedIndexOf(2, 2, 0)}: {interleaved[StridedIndexOf(2, 2, 0)]}");
        Console.WriteLine($"y2 at index {StridedIndexOf(2, 2, 1)}: {interleaved[StridedIndexOf(2, 2, 1)]}");
        Console.WriteLine($"Index 4 -> element {StridedFromIndex(4, 2, 0)} (should be 2)\n");
        
        Console.WriteLine("Test 2: Blocked (x0,x1,x2,x3,y0,y1,y2,y3)");
        int[] blocked = new int[8];
        for (int i = 0; i < 4; i++)
        {
            blocked[BlockIndexOf(i, 4, 0)] = i;
            blocked[BlockIndexOf(i, 4, 1)] = i + 10;
        }
        Console.WriteLine($"Array: [{string.Join(", ", blocked)}]");
        Console.WriteLine($"x2 at index {BlockIndexOf(2, 4, 0)}: {blocked[BlockIndexOf(2, 4, 0)]}");
        Console.WriteLine($"y2 at index {BlockIndexOf(2, 4, 1)}: {blocked[BlockIndexOf(2, 4, 1)]}");
        Console.WriteLine($"Index 6 -> element {BlockFromIndex(6, 4, 1)} in block 1 (should be 2)\n");
        
        Console.WriteLine("Test 3: Padded Strided (stride=2, padding=1)");
        int[] padded = new int[15];
        for (int i = 0; i < 4; i++)
        {
            padded[StridedIndexOf(i, 2, 0, 1)] = i;
            padded[StridedIndexOf(i, 2, 1, 1)] = i + 10; 
        }
        Console.WriteLine($"Array: [{string.Join(", ", padded)}]");
        Console.WriteLine($"x2 at index {StridedIndexOf(2, 2, 0, 1)}: {padded[StridedIndexOf(2, 2, 0, 1)]}");
        
        Console.WriteLine("\nTest 4: Tiled 2D (4x4 matrix, 2x2 tiles)");
        int[] tiled = new int[16];
        for (int row = 0; row < 4; row++)
        for (int col = 0; col < 4; col++)
        {
            int idx = TiledIndexOf(row, col, 2, 2, 4);
            tiled[idx] = row * 4 + col;
        }
        
        Console.WriteLine("Tiled storage order:");
        for (int i = 0; i < 16; i += 4)
            Console.WriteLine($"  [{string.Join(", ", tiled[i..(i+4)].Select(x => x.ToString().PadLeft(2)))}]");
        
        Console.WriteLine("\nRound-trip test:");
        for (int i = 0; i < 16; i++)
        {
            var (row, col) = TiledFromIndex(i, 2, 2, 4);
            int backToIndex = TiledIndexOf(row, col, 2, 2, 4);
            if (backToIndex != i)
                Console.WriteLine($"  FAILED: index {i} -> ({row},{col}) -> {backToIndex}");
        }
        Console.WriteLine("  All round-trips passed!");
    }
}