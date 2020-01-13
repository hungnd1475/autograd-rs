# Experimental tape-based automatic differentiation in Rust

This is an experimental project attempting to implement tape-based automatic differentiation in Rust. The project is heavily inspired by this [article](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation).

Example, suppose we want to compute the gradients of the MSE loss of a 2-layer artificial neural network with sigmoid activation function with respect to the model parameters:

```rust
use autograd_rs::var::{Binary, DotProduct, Scalar, Unary, Var, Vector};
use autograd_rs::{array, Tape};

fn softmax<S>(x: &Var<Vector, S>) -> Var<Vector, Binary> {
    let exp_x = x.exp();
    &exp_x / exp_x.sum()
}

fn loss<S1, S2>(h: &Var<Vector, S1>, y: &Var<Vector, S2>) -> Var<Scalar, Unary> {
    (h - y).pow_const(2.0).sum()
}

fn main() {
    let t = Tape::new();
    let mut x = t.vector_var(2); // the input vector

    let mut w1 = t.matrix_var(3, 2); // the weights of first layer
    let mut b1 = t.vector_var(3); // the bias of first layer
    let z1 = w1.dot(&x) + &b1; // feed forward
    let a1 = softmax(&z1); // activation

    let mut w2 = t.matrix_var(2, 3); // the weights of second layer
    let mut b2 = t.vector_var(2); // the bias of second layer
    let z2 = w2.dot(&a1) + &b2; // feed forward

    let h = softmax(&z2); // activation -> output
    let mut y = t.vector_var(2); // the target vector
    let l = loss(&h, &y); // compute the loss

    x.set(array![1.0, 2.0]);
    y.set(array![1.0, 0.0]);

    w1.set(array![[-1.3, 2.7], [-0.5, 2.1], [3.6, -5.5]]);
    b1.set(array![2.0, -1.5, 0.3]);

    w2.set(array![[-1.0, -0.7, 0.5], [-0.5, 1.8, 1.2]]);
    b2.set(array![0.44, -0.32]);

    println!("l = {}", l.eval());

    let grad = l.grad();
    println!("dl/dw1 =\n {}", grad.wrt(&w1));
    println!("dl/db1 =\n {}", grad.wrt(&b1));

    println!("dl/dw2 =\n {}", grad.wrt(&w2));
    println!("dl/db2 =\n {}", grad.wrt(&b2));
}
```

Output:

```
l = [[0.39630271018754865]]
dl/dw1 =
 [[-0.017104741700976426, -0.03420948340195285],
 [0.01710461381200916, 0.03420922762401832],
 [0.00000012788896732084814, 0.0000002557779346416963]]
dl/db1 =
 [[-0.017104741700976426],
 [0.01710461381200916],
 [0.00000012788896732084814]]
dl/dw2 =
 [[-0.4310572775092538, -0.00872542323941184, -0.0000007977151139850206],
 [0.4310572775092539, 0.00872542323941184, 0.0000007977151139850207]]
dl/db2 =
 [[-0.4397834984637796],
 [0.4397834984637797]]
```
