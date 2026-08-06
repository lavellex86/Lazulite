# CalcContext

```csharp
public partial class CalcContext
```
The context over `Calc32`'s numerical methods.

---

## Constructor

```csharp
public CalcContext(LazuliteContext lctx)
```
Creates a new `CalcContext` under a `LazuliteContext` `lctx`.

---

## Properties

```csharp
public LazuliteContext LContext { get; set; }
```
The Lazulite context over the `CalcContext`.

---

## Methods

```csharp
public RemoteVector ForwardDifferenceStep(RemoteVector fNext, RemoteVector f, float dx)
```
Takes the forward difference of a function using a next and current value.

---

```csharp
public RemoteVector BackwardDifferenceStep(RemoteVector fPrev, RemoteVector f, float dx)
```
Takes the backward difference of a function using a previous and current value.

---

```csharp
public RemoteVector CentralDifferenceStep(RemoteVector fNext, RemoteVector fPrev, float dx)
```
Takes the central difference of a function using a next and previous value.

---

```csharp
public RemoteVector[] Differentiate(RemoteVector[] f, float dx)
```
Takes the derivative of a function.

---

```csharp
public RemoteVector EulerStep(RemoteVector f, RemoteVector prevF, float dx)
```
Takes an Euler integration step using the current function value `f` and previous integral value `prevF`.

---

```csharp
public RemoteVector VerletStep(RemoteVector prevF, RemoteVector prePrevF, RemoteVector prevdf, float dx)
```
Takes a Verlet integration step using the previous integral value `prevF`, pre-previous integral value `prePrevF`, and the previous first-order derivative value `prevdf`.

---

```csharp
public (RemoteVector F, RemoteVector f) VelVerletStep(RemoteVector prevF, RemoteVector prevf, RemoteVector prevdf, RemoteVector df, float dx)
```
Takes a velocity Verlet step using the previous integral value `prevF`, previous function value `prevf`, and previous first-order derivative value `prevdf`.

---

```csharp
public RemoteVector[] EulerIntegrate(RemoteVector[] f, RemoteVector initialF, float dx)
```
Takes the integral of `f` with Euler's method.

---

```csharp
public RemoteVector[] VerletIntegrate(RemoteVector[] df, RemoteVector initialF, RemoteVector initialf, float dx)
```
Takes the integral of a function using its first-order derivative `df` using Verlet's method.

---

```csharp
public (RemoteVector[] F, RemoteVector[] f) VelVerletIntegrate(RemoteVector[] df, RemoteVector initialF, RemoteVector initialf, float dx)
```
Takes the integral of a function using its first-order derivative `df` using the Velocity Verlet method, returning both the integral and the function.