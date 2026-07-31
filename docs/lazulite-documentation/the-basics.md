# The Basics

{% hint style="info" %}
These docs are for readers familiar with the basics of GPU computing. For those who are unfamiliar, the [ILGPU primers](https://ilgpu.net/docs/01-primers/02-a-gpu-is-not-a-cpu/) are a great way to learn.
{% endhint %}

Lazulite operates on the idea of remotes, objects stored on a compute device like the GPU instead of the runtime device, the CPU. Remote types implement the abstract class `RemoteBase<TElement, THost>` , which represents a `THost` type object made up of `TElement`s underneath.

The underlying buffer of `TElements` will belong to a buffer pool, which allows compute device memory to be reused on remote disposal.&#x20;

A simple remote looks like this:

{% code overflow="wrap" %}
```csharp
using ILGPU;
using ILGPU.Runtime;
using Raphael.Lazulite;

// a remote array of integers, we'll take the ILGPU memory buffer and the pool in the constructor
class RemoteIntArray(MemoryBuffer1D<int, Stride1D.Dense> buffer, BufferPool<int> pool) 
    : RemoteBase<int, int[]>(buffer, pool) // then we'll derive from RemoteBase, with TElement = int and THost = int[]
{
    protected override int[] ConvertToHost(int[] raw) => raw; // turns a TElement[] into a THost
    protected override int[] ConvertToRaw(int[] host) => host; // turns a THost into a TElement[]
}
```
{% endcode %}

The `Convert` methods are trivial in this case, because we're storing an array on the compute device; some types have it a little harder, like `float[,]` .

Using `RemoteIntArray` is fairly simple; we'll create one by:

1. Creating a new context object for the pool
2. Retrieving a `int` memory buffer of the target size from the pool
3. Initializing a `RemoteIntArray` with the buffer and pool
4. Using the `.Set(THost host)` method to set the value

In code, this looks like:

{% code overflow="wrap" %}
```csharp
using var lctx = new LazuliteContext(); // creates the context, which will hold everything else
using var pool = new BufferPool<int>(lctx); // creates a pool under the context

using var remote = new RemoteIntArray(pool.Get(3), pool) // takes a length-3 buffer from the buffer pool
    .Set([1, 2, 3]); // sets its value to [1, 2, 3]
```
{% endcode %}

We can't operate on remotes the way we'd operate on a runtime object. Instead, we need to create a kernel for the compute device to run, which is a function runnable in multiple instances or threads. A simple kernel to add `1` to each array element looks like this:

{% code overflow="wrap" %}
```csharp
var kernel = new LazuliteKernel<Action<Index1D, ArrayView1D<int, Stride1D.Dense>>>((i, arr) => arr[i] += 1, lctx); // (i, arr) => arr[i] += 1 runs for each element
kernel.Call(3, remote.Buffer); // we'll call it with an extent of 3 (3 elements) on the remote's bufer
```
{% endcode %}

The `LazuliteKernel` class takes a type parameter equal to the signature of the method it's running, which is always an action starting with an ILGPU `Index1D`. The `ArrayView1D<int, Stride1D.Dense>` is the view into the remote object's memory, which is an array of `int`s.

Now that we've modified our array, we can finally retrieve it with `.Get`:

{% code overflow="wrap" %}
```csharp
var returned = remote.Get(); // takes the object back from the compute device
Console.WriteLine($"[{returned[0]}, {returned[1]}, {returned[2]}]");
```
{% endcode %}

{% hint style="info" %}
You can view the full code for this example [here](https://github.com/raphael286/Lazulite/blob/main/Tests/DocsExample/Program.cs).
{% endhint %}
