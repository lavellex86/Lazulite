# Remotes

{% hint style="info" %}
You can read the [remotebase.md](../lazulite-api-reference/remotebase.md "mention") API reference for more details.
{% endhint %}

On this page, we'll go more in-depth into the specifics of the `RemoteBase` and how to use/implement it effectively.

To begin, let's start with a simple remote type; a color remote:

{% code overflow="wrap" %}
```csharp
using ILGPU;
using ILGPU.Runtime;
using Lavelle.Lazulite;
using System.Drawing;

public class RemoteColor(MemoryBuffer1D<byte, Stride1D.Dense> buffer, BufferPool<byte> pool) : RemoteBase<byte, Color>(buffer, pool) // we'll store 4 0-255 range elements, to make up ARGB in bytes on the compute device
{
    // bytes to ARGB uss the Color.FromArgb method
    protected override Color ConvertToHost(byte[] raw) => Color.FromArgb(raw[0], raw[1], raw[2], raw[3]);
    // ARGB to bytes lists ARGB in raw byte format
    protected override byte[] ConvertToRaw(Color host) => [host.A, host.R, host.G, host.B];
}
```
{% endcode %}

Creating a `RemoteColor` is easy; we'll get grab a length 4 vector from the pool and set it however we like:

<pre class="language-csharp" data-overflow="wrap"><code class="lang-csharp"><strong>using var lctx = new LazuliteContext();
</strong>using var pool = new BufferPool&#x3C;byte>(lctx); // new context and pool

using var red = new RemoteColor(pool.Get(4), pool) // create the remote
    .Set(Color.Red); // set to read
using var blue = new RemoteColor(pool.Get(4), pool) // create the remote
    .Set(Color.Blue); // set to blue
</code></pre>

We can copy the `red` into another remote with `.Set` as well:

{% code overflow="wrap" %}
```csharp
using var redCopy = new RemoteColor(pool.Get(4), pool)
    .Set(red.Buffer); // copies from red on the compute device
```
{% endcode %}

and take the copy's length and dispose status:

{% code overflow="wrap" %}
```csharp
Console.WriteLine($"{redCopy.Length}, {redCopy.Disposed}, {redCopy.Disposable}");
```
{% endcode %}

If a remote is flagged non-disposable, calls to `.Dispose` will be ignored, meaning the buffer will not be returned to the pool.

Standard practice is to leave remotes on the compute device as long as possible; you want to `.Get` the smallest possible result. For example, you might return the sum of elements in an array rather than bringing the array to the CPU, then summing.

{% hint style="info" %}
You can view the full script for this page [here](https://github.com/lavellex86/Lazulite/blob/main/Tests/DocsExample/Program.cs).
{% endhint %}
