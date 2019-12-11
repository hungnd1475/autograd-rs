use autograd_rs::Tape;

fn main() {
    let t = Tape::new();
    let x = t.var(1.0);
    let y = t.var(-2.4);
    let z = x.sin() + x * y - y.pow_const(2.0);

    // println!("tape size = {}", t.len());
    println!("z = {}", z.eval());

    let grad = z.grad();
    println!("dz/dx = {}", grad.wrt(x));
    println!("dz/dy = {}", grad.wrt(y));
}
