use nalgebra::{DVector};

pub fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x| (-x*tau).exp())
}

pub fn exp_decay_dtau(x: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  x.map(|x| -x*(-x * tau).exp() )
}