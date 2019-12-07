# Experimental tape-based automatic differentiation in Rust

This is an experimental project attempting to implement tape-based automatic differentiation in Rust. The project is heavily inspired by this [article](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation). Only some basics operators are currently supported.

Example, compute the gradients of z = sin(x) + xy - y^2 at (x, y) = (1.0, -2.4):

```rust
use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let x = t.var(1.0);
    let y = t.var(-2.4);
    let z = x.sin() + x * y - y.pow_const(2.0);

    println!("z = {}", z.eval()); // -7.31853

    let grad = z.grad();
    println!("dz/dx = {}", grad.wrt(x)); // -1.8597
    println!("dz/dy = {}", grad.wrt(y)); // 5.8
}
```
