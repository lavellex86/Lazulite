using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public partial class KernelProgramming
{
    public static int StridedIndexOf(int i, int stride, int offset = 0, int padding = 0) => i * (stride + padding) + offset;
    public static int StridedFromIndex(int index, int stride, int offset = 0, int padding = 0) => (index - offset) / (stride + padding);
    
    public static int BlockIndexOf(int i, int blockSize, int blockIndex) => blockIndex * blockSize + i;
    public static int BlockFromIndex(int index, int blockSize, int blockIndex) => index - blockIndex * blockSize;
    
    public static int CyclicIndexOf(int i, int size) => i % size;
    public static int CyclicCycleCount(int i, int size) => i / size;
    
    public static int TiledIndexOf(int row, int col, int tileRows, int tileCols, int matrixCols)
    {
        int tileRow = row / tileRows;
        int tileCol = col / tileCols;
        int inTileRow = row % tileRows;
        int inTileCol = col % tileCols;
    
        int tilesPerRow = matrixCols / tileCols;
        int tileStart = (tileRow * tilesPerRow + tileCol) * (tileRows * tileCols);
        return tileStart + inTileRow * tileCols + inTileCol;
    }

    public static (int row, int col) TiledFromIndex(int index, int tileRows, int tileCols, int matrixCols)
    {
        int tilesPerRow = matrixCols / tileCols;
        int tileSize = tileRows * tileCols;
    
        int tileIndex = index / tileSize;
        int inTileIndex = index % tileSize;
    
        int tileRow = tileIndex / tilesPerRow;
        int tileCol = tileIndex % tilesPerRow;
    
        int inTileRow = inTileIndex / tileCols;
        int inTileCol = inTileIndex % tileCols;
    
        return (tileRow * tileRows + inTileRow, tileCol * tileCols + inTileCol);
    }
}