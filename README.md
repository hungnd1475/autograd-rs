# Experimental tape-based automatic differentiation in Rust

This is an experimental project attempting to implement tape-based automatic differentiation in Rust. The project is heavily inspired by this [article](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation).

Example, suppose we want to compute the gradients of the MSE loss of a 2-layer artificial neural network with sigmoid activation function with respect to the model parameters:

```rust
use autograd_rs::{array, ScalarVar, Tape, VectorVar};

fn sigmoid(x: &VectorVar) -> VectorVar {
    1.0 / (1.0 + (-x).exp())
}

fn loss(h: &VectorVar, y: &VectorVar) -> ScalarVar {
    (h - y).pow_const(2.0).sum()
}

fn main() {
    let t = Tape::new();
    let mut x = t.vector_var(2); // the input vector

    let mut w1 = t.matrix_var(3, 2); // the weights of first layer
    let mut b1 = t.vector_var(3); // the bias of first layer
    let z1 = w1.dot(&x) + &b1; // feed forward
    let a1 = sigmoid(&z1); // activation

    let mut w2 = t.matrix_var(2, 3); // the weights of second layer
    let mut b2 = t.vector_var(2); // the bias of second layer
    let z2 = w2.dot(&a1) + &b2; // feed forward

    let h = sigmoid(&z2); // activation -> output
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
l = [[1.0640323060200898]]
dl/dw1 =
 [[0.00028334741133117575, 0.0005666948226623515],
 [0.06495812513327018, 0.12991625026654036],
 [0.000178690494143982, 0.000357380988287964]]
dl/db1 =
 [[0.00028334741133117575],
 [0.06495812513327018],
 [0.000178690494143982]]
dl/dw2 =
 [[-0.27385129545037956, -0.24708743904323366, -0.00022627614009696203],
 [0.2944706061285532, 0.2656915966826116, 0.0002433133354991017]]
dl/db2 =
 [[-0.2744655076808846],
 [0.29513106474537637]]
```
