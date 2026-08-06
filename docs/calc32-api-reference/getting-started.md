# Getting Started

To begin, create a `CalcContext` object with your `LazuliteContext`:
```csharp
var lctx = new LazuliteContext();
var cctx = new CalcContext(lctx); // new CalcContext for computation
```
Then call whatever methods you need from it. For example, to differentate `[1, 2, 3, 4]` (a linear relation with some spacing `dx`)
```csharp
var f1 = new RemoteVector[]
{
    lctx.GetVector(1).Set([1]).AsVector(), // get a vector, set it to the output at each timestep
    lctx.GetVector(1).Set([2]).AsVector(),
    lctx.GetVector(1).Set([3]).AsVector(),
    lctx.GetVector(1).Set([4]).AsVector()
};
var df1 = cctx.Differentiate(f1, 0.01f); // differentiate with dx = 0.01f, 
```
Both differentiation and integration operations take functions in the form `RemoteVector[]` for vector functions; the array represnts the function over `x`, the vector represnts the output. 
We could integrate a vector function:
```csharp
var f2 = new RemoteVector[]
{
    lctx.GetVector(3).Set([1, 1, 1]).AsVector(),
    lctx.GetVector(3).Set([2, 2, 2]).AsVector(),
    lctx.GetVector(3).Set([1, 1, 1]).AsVector()
};
var initialF2 = lctx.GetVector(3).Set([0, 0, 0]).AsVector(); // set F(0)
var F2 = cctx.EulerIntegrate(f2, initialF2, 0.01f); // integrate with dx = 0.01f
```
with `F2` as the integral (or antiderivative) of `f2`. Methods that refer to `df` are referring to the first derivative of the function, or the second derivative of the integral.