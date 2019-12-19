use autograd_rs::Tape;
use ndarray::Array2;

fn main() {
    let t = Tape::new();
    let x = t.vector_var(3);
    let w = t.matrix_var(2, 3);
    let b = t.vector_var(2);
    let h = w.dot(x) + b;
    let z = h.sin();

    let w2 = t.vector_var(2);
    let b2 = t.scalar_var();
    let z2 = w2.t().dot(z) + b2;

    x.set(vec![1.0, 2.0, 3.0]);
    w.set(vec![2.1, -4.5, 3.7, 1.9, -4.8, 5.1]);
    b.set(vec![-4.5, -3.2]);

    w2.set(vec![0.3, -1.2]);
    b2.set(vec![-4.4]);

    println!("z = {}", z2.eval());

    let grad = z2.grad();
    println!("dz/dx = {}", grad.wrt(x));
    println!("dz/dw = {}", grad.wrt(w));
    println!("dz/db = {}", grad.wrt(b));
}
