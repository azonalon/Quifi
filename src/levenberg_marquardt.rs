use levenberg_marquardt::{LeastSquaresProblem};
use nalgebra::{DVector, DMatrix, storage::Owned, Dynamic};
use makima_spline::Spline;

fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x| (-x*tau).exp())
}

fn exp_decay_dtau(x: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  x.map(|x| -x*(-x * tau).exp())
}

#[derive(Clone, PartialEq, Debug)]
pub struct NExponentialProblem {
    // holds current value of the n parameters
    pub x: DVector<f64>,
    pub y: DVector<f64>,
    pub p: DVector<f64>, // parameters
}

pub struct InterpolationTranslationProblem {
    // holds current value of the n parameters
    pub x: DVector<f64>,
    pub y: DVector<f64>,
    pub p: DVector<f64>, // parameters
    pub spline: Spline,
}

impl LeastSquaresProblem<f64, Dynamic, Dynamic> for NExponentialProblem {
    type ParameterStorage = Owned<f64, Dynamic>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
     
    fn set_params(&mut self, p: &DVector<f64>) {
        std::assert!(p.len() >= 3 && (p.len() % 2 == 1), "Parameter count {} is invalid", p.len());
        self.p.copy_from(p);
        // do common calculations for residuals and the Jacobian here
    }
     
    fn params(&self) -> DVector<f64> { self.p.clone() }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut r = - self.p[0]*DVector::<f64>::from_element(self.x.len(), 1.0) - self.y.clone();
        for i in 0..(self.p.len()-1)/2  {
            r += self.p[1+2*i]*exp_decay(&self.x, self.p[2+2*i]); 
        }
        Some(r)
    }
     
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let mut columns=Vec::<DVector<f64>>::new();
        columns.push(-DVector::<f64>::from_element(self.x.len(), 1.0));
        for i in 0..(self.p.len()-1)/2  {
            columns.push(exp_decay(&self.x, self.p[2*(i+1)]));
            columns.push(exp_decay_dtau(&self.x, self.p[2*(i+1)]));
        }
        Some(DMatrix::<f64>::from_columns(columns.as_slice()))
    }
}

impl LeastSquaresProblem<f64, Dynamic, Dynamic> for InterpolationTranslationProblem {
    type ParameterStorage = Owned<f64, Dynamic>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
     
    fn set_params(&mut self, p: &DVector<f64>) {
        std::assert!(p.len() >= 3 && (p.len() % 2 == 1), "Parameter count {} is invalid", p.len());
        self.p.copy_from(p);
        // do common calculations for residuals and the Jacobian here
    }
     
    fn params(&self) -> DVector<f64> { self.p.clone() }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut r = - self.p[0]*DVector::<f64>::from_element(self.x.len(), 1.0) - self.y.clone();
        for i in 0..(self.p.len()-1)/2  {
            r += self.p[1+2*i]*exp_decay(&self.x, self.p[2+2*i]); 
        }
        Some(r)
    }
     
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let mut columns=Vec::<DVector<f64>>::new();
        columns.push(-DVector::<f64>::from_element(self.x.len(), 1.0));
        for i in 0..(self.p.len()-1)/2  {
            columns.push(exp_decay(&self.x, self.p[2*(i+1)]));
            columns.push(exp_decay_dtau(&self.x, self.p[2*(i+1)]));
        }
        Some(DMatrix::<f64>::from_columns(columns.as_slice()))
    }
}