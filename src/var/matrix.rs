use super::{Var, VectorVar};
use crate::{BinaryOp, Matrix, Shape, Tape, VarNode};

#[derive(Clone, Copy)]
pub struct MatrixVar<'t> {
    tape: &'t Tape,
    shape: Shape,
    index: usize,
}

impl<'t> Var for MatrixVar<'t> {
    fn index(&self) -> usize {
        self.index
    }
}

impl<'t> MatrixVar<'t> {
    pub(crate) fn new(tape: &'t Tape, shape: Shape, index: usize) -> Self {
        Self { tape, shape, index }
    }

    /// Sets the value of the variable.
    pub fn set(&mut self, new_value: Vec<f64>) {
        let new_value = Matrix::from_shape_vec(self.shape, new_value).expect(&format!(
            "The given value cannot be coerced into a matrix of shape {:?}.",
            self.shape
        ));
        let mut vars = self.tape.var_nodes.borrow_mut();
        match &mut vars[self.index] {
            VarNode::Nullary { ref mut value, .. } => *value = Some(new_value),
            _ => panic!("Cannot set value for non-input variable."),
        }
        self.tape.is_evaluated.set(false);
    }

    pub fn dot(&self, other_vector: VectorVar<'t>) -> VectorVar<'t> {
        let op = BinaryOp::Dot;
        let shape = op.eval_shape(self.shape, other_vector.shape());
        VectorVar::new(
            self.tape,
            shape,
            self.tape
                .push_binary(shape, self.index, other_vector.index(), op),
        )
    }
}
