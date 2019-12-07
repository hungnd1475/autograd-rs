use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let x = t.var(1.0);
    let y = t.var(-2.4);
    let z = x.sin() + x * y - y.pow_const(2.0);

    println!("{}", z.eval());

    let grad = z.grad();
    println!("{}", grad.wrt(y));

    x.set(2.0);
    println!("{}", z.eval());

    let grad = z.grad();
    println!("{}", grad.wrt(x));
}
