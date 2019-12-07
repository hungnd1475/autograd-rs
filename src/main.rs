use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let x = t.var();
    let y = t.var();
    let z = x * y + x.pow(y);

    x.set_value(1.0);
    y.set_value(-2.4);
    println!("{}", z.eval());

    let grad = z.grad();
    println!("{}", grad.wrt(y));
}
