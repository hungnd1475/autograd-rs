pub mod activate;
mod op;
mod tape;
pub mod var;

pub use ndarray::array;
pub use tape::Tape;

use ndarray::{Array1, Array2};

pub type FloatVector = Array1<f64>;
pub type FloatMatrix = Array2<f64>;
