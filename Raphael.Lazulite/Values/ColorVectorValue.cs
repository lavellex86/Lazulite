using System.Drawing;
using ILGPU;
using ILGPU.Runtime;

namespace Raphael.Lazulite;

public class ColorVectorValue : Value<,>
{
    public ColorVectorValue(Color[] value, int aidx) : base(Compute.Get(aidx, value.Length), [value.Length]) => FromHost(value);
    public ColorVectorValue(MemoryBuffer1D<float, Stride1D.Dense> buffer) : base(buffer, [(int)buffer.Length]) { }

    public override Color[] Unroll(float[] rolled) => ColorVectorProxy.Unroll(rolled);
    public override float[] Roll(Color[] value) => ColorVectorProxy.Roll(value);
    public override ColorVectorValue Create(MemoryBuffer1D<float, Stride1D.Dense> buffer, int[] shape) => new(buffer);
    public override ColorVectorProxy ToProxy() => new(this);
}

public class ColorVectorProxy(ColorVectorValue value) : ValueProxy<Color[]>(value)
{
    public override float Get(int[] index) => FlatData[index[0]];
    public override Color[] ToHost() => Unroll(FlatData);

    public static float ColorToFloat(Color color)
    {
        var (a, r, g, b) = (color.A, color.R, color.G, color.B);
        int colorInt = (a << 24) | (r << 16) | (g << 8) | b;
        return BitConverter.Int32BitsToSingle(colorInt);
    }
    public static Color FloatToColor(float value)
    {
        int colorInt = BitConverter.SingleToInt32Bits(value);
        byte a = (byte)((colorInt >> 24) & 0xFF);
        byte r = (byte)((colorInt >> 16) & 0xFF);
        byte g = (byte)((colorInt >> 8) & 0xFF);
        byte b = (byte)(colorInt & 0xFF);
        return Color.FromArgb(a, r, g, b);
    }
    
    public static float[] Roll(Color[] value) => value.Select(ColorToFloat).ToArray();
    public static Color[] Unroll(float[] rolled) => rolled.Select(FloatToColor).ToArray();
}