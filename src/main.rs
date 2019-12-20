use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let mut w = t.matrix_var(2, 3);
    let mut x = t.vector_var(3);
    let mut b = t.vector_var(2);
    let mut y = t.vector_var(2);
    let f = w.dot(x) + b;
    let l = (f - y).l2norm();

    x.set(vec![1.0, 2.0, 3.0]);
    w.set(vec![-1.3, 2.7, -0.5, 2.1, 3.6, -5.5]);
    b.set(vec![2.0, -1.5]);
    y.set(vec![0.0, 1.0]);
    println!("z = {}", l.eval());

    let grad = l.grad();
    println!("dl/dw =\n {}", grad.wrt(&w));
    println!("dl/db =\n {}", grad.wrt(&b));
}
