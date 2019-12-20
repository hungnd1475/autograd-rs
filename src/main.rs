use autograd_rs::{ScalarVar, Tape, VectorVar};

fn sigmoid<'t>(x: VectorVar<'t>) -> VectorVar<'t> {
    1.0 / (1.0 + (-x).exp())
}

fn loss<'t>(h: VectorVar<'t>, y: VectorVar<'t>) -> ScalarVar<'t> {
    (h - y).pow_const(2.0).sum()
}

fn main() {
    let t = Tape::new();
    let mut x = t.vector_var(2);

    let mut w1 = t.matrix_var(3, 2);
    let mut b1 = t.vector_var(3);
    let z1 = w1.dot(x) + b1;
    let a1 = sigmoid(z1);

    let mut w2 = t.matrix_var(1, 3);
    let mut b2 = t.vector_var(1);
    let z2 = w2.dot(a1) + b2;

    let h = sigmoid(z2);
    let mut y = t.vector_var(1);
    let l = loss(h, y);

    x.set(vec![1.0, 2.0]);
    y.set(vec![1.0]);

    w1.set(vec![-1.3, 2.7, -0.5, 2.1, 3.6, -5.5]);
    b1.set(vec![2.0, -1.5, 0.3]);

    w2.set(vec![-1.0, -0.7, 0.5]);
    b2.set(vec![0.44]);

    println!("l = {}", l.eval());

    let grad = l.grad();
    println!("dl/dw1 =\n {}", grad.wrt(&w1));
    println!("dl/db1 =\n {}", grad.wrt(&b1));

    println!("dl/dw2 =\n {}", grad.wrt(&w2));
    println!("dl/db2 =\n {}", grad.wrt(&b2));
}
