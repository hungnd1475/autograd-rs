impl<SL, SR> Div<&Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Scalar, SR>> for &Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: &Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Scalar, SR>> for Var<Vector, SL> {
    type Output = Var<Vector, Binary>;

    fn div(self, other: Var<Scalar, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Vector, SR>> for &Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<&Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: &Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}

impl<SL, SR> Div<Var<Vector, SR>> for Var<Matrix, SL> {
    type Output = Var<Matrix, Binary>;

    fn div(self, other: Var<Vector, SR>) -> Self::Output {
        self.binary(&other.broadcast(self.kind.shape()), BinaryOp::Div)
    }
}
