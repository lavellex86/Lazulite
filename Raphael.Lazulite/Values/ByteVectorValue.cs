using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class ByteVectorValue : Value<,>
{
    public int OriginalLength { get; private set; }
    
    public ByteVectorValue(byte[] value, int aidx) : base(Compute.Get(aidx, (value.Length + 3) / 4), [(value.Length + 3) / 4])
    {
        OriginalLength = value.Length;
        FromHost(value);
    }
    
    public ByteVectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer, int originalLength) : base(buffer, [(int)buffer.Length]) => OriginalLength = originalLength;

    public override byte[] Unroll(float[] rolled) => ByteVectorProxy.Unroll(rolled, OriginalLength);
    public override float[] Roll(byte[] value) => ByteVectorProxy.Roll(value);
    public override ByteVectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer, OriginalLength);
    public override ByteVectorProxy ToProxy() => new(this);
}

public class ByteVectorProxy(ByteVectorValue value) : ValueProxy<byte[]>(value)
{
    private int OriginalLength => value.OriginalLength;
    
    public override float Get(int[] index) => FlatData[index[0]];
    public override byte[] ToHost() => Unroll(FlatData, OriginalLength);
    
    public static float BytesToFloat(byte b0, byte b1, byte b2, byte b3, bool bigEndian = true)
    {
        int packed;
        if (bigEndian) packed = b0 << 24 | b1 << 16 | b2 << 8 | b3;
        else packed = b0 | b1 << 8 | b2 << 16 | b3 << 24;
        return Interop.FloatAsInt(packed);
    }

    public static (byte, byte, byte, byte) FloatToBytes(float value, bool bigEndian = true)
    {
        var packed = Interop.FloatAsInt(value);
    
        byte b0, b1, b2, b3;
        if (bigEndian)
        {
            b0 = (byte)(packed >> 24 & 0xFF);
            b1 = (byte)(packed >> 16 & 0xFF);
            b2 = (byte)(packed >> 8 & 0xFF);
            b3 = (byte)(packed & 0xFF);
        }
        else
        {
            b3 = (byte)(packed >> 24 & 0xFF);
            b2 = (byte)(packed >> 16 & 0xFF);
            b1 = (byte)(packed >> 8 & 0xFF);
            b0 = (byte)(packed & 0xFF);
        }
    
        return (b0, b1, b2, b3);
    }
    
    public static float[] Roll(byte[] value, bool bigEndian = true)
    {
        var floatCount = (value.Length + 3) / 4;
        var result = new float[floatCount];
    
        for (int i = 0; i < floatCount; i++)
        {
            int baseIdx = i * 4;
            byte b0 = baseIdx < value.Length ? value[baseIdx] : (byte)0;
            byte b1 = baseIdx + 1 < value.Length ? value[baseIdx + 1] : (byte)0;
            byte b2 = baseIdx + 2 < value.Length ? value[baseIdx + 2] : (byte)0;
            byte b3 = baseIdx + 3 < value.Length ? value[baseIdx + 3] : (byte)0;
        
            result[i] = BytesToFloat(b0, b1, b2, b3, bigEndian);
        }
    
        return result;
    }

    public static byte[] Unroll(float[] rolled, int originalLength, bool bigEndian = true)
    {
        byte[] result = new byte[originalLength];
        int byteIndex = 0;
    
        foreach (float f in rolled)
        {
            if (byteIndex >= originalLength) break;
        
            var (b0, b1, b2, b3) = FloatToBytes(f, bigEndian);
        
            if (byteIndex < originalLength) result[byteIndex++] = b0;
            if (byteIndex < originalLength) result[byteIndex++] = b1;
            if (byteIndex < originalLength) result[byteIndex++] = b2;
            if (byteIndex < originalLength) result[byteIndex++] = b3;
        }
    
        return result;
    }
}