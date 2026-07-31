# Setting Up a Context

{% hint style="info" %}
You can read the[lazulitecontext.md](../lazulite-api-reference/lazulitecontext.md "mention")API reference for more details.
{% endhint %}

The `LazuliteContext` object is very important to set up correctly. On this page, we'll walk through it's usage and customization.

The constructor for `LazuliteContext` includes two options we haven't used yet; a `gpu` boolean, which sets whether the initializer will look for a CPU or GPU compute device, and the `optimization` enum, which determines how fast kernels are after compilation.

<pre class="language-csharp" data-overflow="wrap"><code class="lang-csharp">using ILGPU;
using Raphael.Lazulite;

using var lctx1 = new LazuliteContext(gpu: true, optimization: OptimizationLevel.Debug); // uses GPU, debug optimization; fast compliation, slower kernels
using var lctx2 = new LazuliteContext(false, OptimizationLevel.O0); // uses CPU, optimization level 0
using var lctx3 = new LazuliteContext(true, OptimizationLevel.O1); // uses GPU, optimization level 1
using var lctx4 = new LazuliteContext(false, OptimizationLevel.O2);// uses CPU, optimization level 2
<strong>using var lctx5 = new LazuliteContext(true, OptimizationLevel.Release); // uses GPU, release optimization; longer kernel compilation for faster kernels - this is the default configuration
</strong></code></pre>

If you're using any extension libraries for Lazulite, this is where you would initialize them. In general, the enabling methods will look like this:

{% code overflow="wrap" %}
```csharp
using Raphael.Linalg32;

using var lctx = new LazuliteContext()
    .EnableLinalg32()
```
{% endcode %}

Once that's done, the context is mostly used to manage everything underneath it and dispose of it all at once. If you have an action you'd like to run on context disposal, add it too `.DisposeHooks`:

{% code overflow="wrap" %}
```csharp
lctx.DisposeHooks.Add(() => Console.WriteLine("Disposing!"));
```
{% endcode %}

You can also manually synchronize the compute and runtime devices with `.Synchronize`:

{% code overflow="wrap" %}
```csharp
lctx.Synchronize();
```
{% endcode %}

and quickly grab the name of the compute device with `.AcceleratorName`, along with the ILGPU accelerator from `.Accelerator`:

{% code overflow="wrap" %}
```csharp
Console.WriteLine(lctx.AcceleratorName);
Console.WriteLine(lctx.Accelerator.AcceleratorType);
```
{% endcode %}

{% hint style="info" %}
You can view the full script for this page [here](https://github.com/raphael286/Lazulite/blob/main/Tests/DocsExample/Program.cs).
{% endhint %}
