use crate::tape::Node;
use crate::var::Shape;
use crate::FloatMatrix;
use ndarray::{stack, Axis, Zip};

#[derive(Clone, Copy, Debug, PartialEq)]
/// Represents the supported unary operations.
pub(crate) enum UnaryOp {
    T,
    Neg,
    Sin,
    Cos,
    Tan,
    Ln,
    Exp,
    Sum(usize),
    Broadcast(Shape),
}

impl UnaryOp {
    pub(crate) fn eval_shape(&self, input_shape: Shape) -> Shape {
        match (self, input_shape) {
            (UnaryOp::T, Shape([nrow, ncol])) => Shape([ncol, nrow]),
            (UnaryOp::Sum(axis), _) => {
                let Shape(mut shape) = input_shape;
                shape[*axis] = 1;
                Shape(shape)
            }
            (UnaryOp::Broadcast(shape), _) => *shape,
            _ => input_shape,
        }
    }

    /// Evaluates the operation with the given parameter.
    pub(crate) fn eval(&self, value: &FloatMatrix) -> FloatMatrix {
        match self {
            UnaryOp::T => value.t().to_owned(),
            UnaryOp::Neg => -value,
            UnaryOp::Sin => sin(value),
            UnaryOp::Cos => cos(value),
            UnaryOp::Tan => tan(value),
            UnaryOp::Ln => ln(value),
            UnaryOp::Exp => exp(value),
            UnaryOp::Sum(axis) => {
                let shape = self.eval_shape(value.dim().into());
                value.sum_axis(Axis(*axis)).into_shape(shape.dim()).unwrap()
            }
            UnaryOp::Broadcast(shape) => value.broadcast(shape.dim()).unwrap().to_owned(),
        }
    }

    /// Computes the gradient of the operation with respect to the given parameter.
    pub(crate) fn grad(&self, node: &Node, ans: &FloatMatrix, g: &FloatMatrix) -> FloatMatrix {
        let value = node.value();
        match node {
            Node::Constant(value) => zeros_like(value),
            _ => match self {
                UnaryOp::T => g.t().to_owned(),
                UnaryOp::Neg => -g,
                UnaryOp::Sin => cos(value) * g,
                UnaryOp::Cos => -sin(value) * g,
                UnaryOp::Tan => 2.0 * g / (cos(&(2.0 * value)) + 1.0),
                UnaryOp::Ln => g / value,
                UnaryOp::Exp => ans * g,
                UnaryOp::Sum(axis) => {
                    let Shape(shape) = node.shape();
                    stack(Axis(*axis), &vec![g.view(); shape[*axis]]).unwrap()
                }
                UnaryOp::Broadcast(_) => {
                    let result = g.sum_axis(Axis(0));
                    let length = result.len();
                    result.into_shape((1, length)).unwrap()
                }
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(dead_code)]
/// Represents the supported binary operations.
pub(crate) enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Log,
    Dot,
    Max,
    Min,
}

impl BinaryOp {
    pub(crate) fn eval_shape(&self, left_shape: Shape, right_shape: Shape) -> Shape {
        match self {
            BinaryOp::Dot => {
                let Shape([left_row, _]) = left_shape;
                let Shape([_, right_col]) = right_shape;
                Shape([left_row, right_col])
            }
            _ => left_shape,
        }
    }

    /// Evaluates the operation with the given parameters.
    pub(crate) fn eval(&self, left: &FloatMatrix, right: &FloatMatrix) -> FloatMatrix {
        match self {
            BinaryOp::Add => left + right,
            BinaryOp::Sub => left - right,
            BinaryOp::Mul => left * right,
            BinaryOp::Div => left / right,
            BinaryOp::Pow => pow(left, right),
            BinaryOp::Log => log(left, right),
            BinaryOp::Dot => left.dot(right),
            BinaryOp::Max => {
                let mut result = left.clone();
                result.zip_mut_with(right, |x, &y| {
                    if *x < y {
                        *x = y
                    }
                });
                result
            }
            BinaryOp::Min => {
                let mut result = left.clone();
                result.zip_mut_with(right, |x, &y| {
                    if *x > y {
                        *x = y
                    }
                });
                result
            }
        }
    }

    /// Computes the partial gradient of the operation with repsect to the left parameter.
    fn left_grad(
        &self,
        left: &FloatMatrix,
        right: &FloatMatrix,
        ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> FloatMatrix {
        match self {
            BinaryOp::Add => g.clone(),
            BinaryOp::Sub => g.clone(),
            BinaryOp::Mul => right * g,
            BinaryOp::Div => g / right,
            BinaryOp::Pow => right * &pow(left, &(right - 1.0)) * g,
            BinaryOp::Log => g / &(left * &ln(right)),
            BinaryOp::Dot => g.dot(&right.t()),
            BinaryOp::Max => balanced_eq(left, right, ans),
            BinaryOp::Min => balanced_eq(left, right, ans),
        }
    }

    /// Computes the partial gradient of the operation with repsect to the right parameter.
    fn right_grad(
        &self,
        left: &FloatMatrix,
        right: &FloatMatrix,
        ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> FloatMatrix {
        match self {
            BinaryOp::Add => g.clone(),
            BinaryOp::Sub => -g,
            BinaryOp::Mul => left * g,
            BinaryOp::Div => -left * g / pow_scalar(right, 2.0),
            BinaryOp::Pow => ln(left) * pow(left, right) * g,
            BinaryOp::Log => -ln(left) * g / (pow_scalar(&ln(right), 2.0) * right),
            BinaryOp::Dot => left.t().dot(g),
            BinaryOp::Max => balanced_eq(left, right, ans),
            BinaryOp::Min => balanced_eq(right, left, ans),
        }
    }

    /// Computes the full gradient of the operation with repsect to the given parameters.
    pub(crate) fn grad(
        &self,
        left_node: &Node,
        right_node: &Node,
        ans: &FloatMatrix,
        g: &FloatMatrix,
    ) -> [FloatMatrix; 2] {
        let left_val = left_node.value();
        let right_val = right_node.value();

        let lg = {
            match left_node {
                Node::Constant(value) => zeros_like(value),
                _ => self.left_grad(left_val, right_val, ans, g),
            }
        };

        let rg = {
            match right_node {
                Node::Constant(value) => zeros_like(value),
                _ => self.right_grad(left_val, right_val, ans, g),
            }
        };

        [lg, rg]
    }
}

fn zeros_like(x: &FloatMatrix) -> FloatMatrix {
    FloatMatrix::zeros(x.dim())
}

fn sin(x: &FloatMatrix) -> FloatMatrix {
    x.mapv(|x| x.sin())
}

fn cos(x: &FloatMatrix) -> FloatMatrix {
    x.mapv(|x| x.cos())
}

fn tan(x: &FloatMatrix) -> FloatMatrix {
    x.mapv(|x| x.tan())
}

fn ln(x: &FloatMatrix) -> FloatMatrix {
    x.mapv(|x| x.ln())
}

fn exp(x: &FloatMatrix) -> FloatMatrix {
    x.mapv(|x| x.exp())
}

fn pow(x: &FloatMatrix, y: &FloatMatrix) -> FloatMatrix {
    let mut result = x.clone();
    result.zip_mut_with(y, |x, &y| *x = x.powf(y));
    result
}

fn pow_scalar(x: &FloatMatrix, p: f64) -> FloatMatrix {
    x.mapv(|x| x.powf(p))
}

fn log(x: &FloatMatrix, y: &FloatMatrix) -> FloatMatrix {
    let mut result = x.clone();
    result.zip_mut_with(y, |x, &y| *x = x.log(y));
    result
}

fn elementwise_eq(x: &FloatMatrix, y: &FloatMatrix) -> FloatMatrix {
    let mut result = FloatMatrix::from_elem(x.dim(), 0.0);
    Zip::from(&mut result)
        .and(x)
        .and(y)
        .apply(|a, &b, &c| *a = if b == c { 1.0 } else { 0.0 });
    result
}

fn balanced_eq(x: &FloatMatrix, y: &FloatMatrix, z: &FloatMatrix) -> FloatMatrix {
    let ones = FloatMatrix::ones(x.dim());
    elementwise_eq(x, y) / (ones + elementwise_eq(x, z))
}
