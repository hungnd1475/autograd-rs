use ndarray::{Array1, Array2};

pub type FloatVector = Array1<f64>;
pub type FloatMatrix = Array2<f64>;

pub trait MatrixFunc {
    fn zeros_like(&self) -> FloatMatrix;
    fn ones_like(&self) -> FloatMatrix;

    fn sin(&self) -> FloatMatrix;
    fn cos(&self) -> FloatMatrix;
    fn tan(&self) -> FloatMatrix;
    fn ln(&self) -> FloatMatrix;
    fn exp(&self) -> FloatMatrix;

    fn pow(&self, power: &FloatMatrix) -> FloatMatrix;
    fn pow_scalar(&self, p: f64) -> FloatMatrix;
    fn log(&self, base: &FloatMatrix) -> FloatMatrix;
}

impl MatrixFunc for FloatMatrix {
    fn zeros_like(&self) -> FloatMatrix {
        FloatMatrix::zeros(self.dim())
    }

    fn ones_like(&self) -> FloatMatrix {
        FloatMatrix::ones(self.dim())
    }

    fn sin(&self) -> FloatMatrix {
        self.mapv(|x| x.sin())
    }

    fn cos(&self) -> FloatMatrix {
        self.mapv(|x| x.cos())
    }

    fn tan(&self) -> FloatMatrix {
        self.mapv(|x| x.tan())
    }

    fn ln(&self) -> FloatMatrix {
        self.mapv(|x| x.ln())
    }

    fn exp(&self) -> FloatMatrix {
        self.mapv(|x| x.exp())
    }

    fn pow(&self, power: &FloatMatrix) -> FloatMatrix {
        let mut result = self.clone();
        result.zip_mut_with(power, |x, y| *x = x.powf(*y));
        result
    }

    fn pow_scalar(&self, p: f64) -> FloatMatrix {
        let power = FloatMatrix::from_elem(self.dim(), p);
        self.pow(&power)
    }

    fn log(&self, base: &FloatMatrix) -> FloatMatrix {
        let mut result = self.clone();
        result.zip_mut_with(base, |x, y| *x = x.log(*y));
        result
    }
}
