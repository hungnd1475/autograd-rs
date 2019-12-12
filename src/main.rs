use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let x = t.var(vec![1.0, 2.0, 3.0]);
    let y = t.var(vec![1.0, -1.5, 3.2]);
    let z = x.sin() + x * y - y.pow_const(2.0);

    println!("z = {}", z.eval());

    let grad = z.grad();
    println!("dz/dx = {}", grad.wrt(x));
    println!("dz/dy = {}", grad.wrt(y));
}
