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
    pub fn set(&mut self, value: Vec<f64>) {
        let value = Matrix::from_shape_vec(self.shape, value).expect(&format!(
            "The given value cannot be coerced into a matrix of shape {:?}.",
            self.shape
        ));
        let vars = self.tape.var_nodes.borrow();
        let mut vals = self.tape.var_values.borrow_mut();
        match &vars[self.index] {
            VarNode::Nullary(_) => vals[self.index] = Some(value),
            _ => panic!("Cannot set value for non-input variable."),
        }
        // invalidate the tape
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
