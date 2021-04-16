extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2,s};
use ndarray_npy::ReadNpyExt;
use nalgebra::{DVector,DMatrix};
use varpro::prelude::*;
use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};

fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x|(-x*tau).exp())
}

fn exp_decay_dtau(tvec: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  tvec.map(|t| -t*((-t * tau).exp()))
}

pub fn fit_triexponential(x: DVector<f64>, y: DVector<f64>,init_values: &[f64]) -> (f64, DVector<f64>) {
    let model =   SeparableModelBuilder::<f64>::new(&["tau1", "tau2", "tau3"])
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .function(&["tau3"], exp_decay)
        .partial_deriv("tau3", exp_decay_dtau)
        .invariant_function(|x|DVector::from_element(x.len(),1.))
        .build()
        .expect("double exponential model builder should produce a valid model");
    // 2. Cast the fitting problem as a nonlinear least squares minimization problem
    let problem = LevMarProblemBuilder::new()
        .model(&model)
        .x(x)
        .y(y)
        // .epsilon(0.00000)
        .initial_guess(init_values)
        .build()
        .expect("Building valid problem should not panic");
    // 3. Solve the fitting problem
    println!("minimize...");
    let user_tol = 1e-5_f64;
    let (solved_problem, report) = LevMarSolver::new()
        .with_ftol(user_tol)
        .with_gtol(user_tol)
        .with_patience(100)
        .with_stepbound(1.0)
        .minimize(problem);
    println!("minimized...");
    assert!(report.termination.was_successful());
    // 4. obtain the nonlinear parameters after fitting
    // let alpha = solved_problem.params();
    let alpha = solved_problem.params();
    let c = solved_problem.linear_coefficients().unwrap();
    // println!("nfev={}, f(x)={}", report.number_of_evaluations, report.objective_function);
    // let yf = model.eval(&x, alpha.as_slice()).unwrap()*c.clone();
    // vec![alpha[0], alpha[1], c[0], c[1], c[2]]
    // let v = Vec::<f64>::new();
    let p = DVector::<f64>::from_vec(vec![alpha[0], alpha[1], alpha[2], c[0], c[1], c[2], c[3]]);
    (report.objective_function, p)
}
pub fn fit_biexponential(x: DVector<f64>, y: DVector<f64>,init_values: &[f64]) -> (f64, DVector<f64>) {
    let model =   SeparableModelBuilder::<f64>::new(&["tau1", "tau2"])
        .function(&["tau1"], exp_decay)
        .partial_deriv("tau1", exp_decay_dtau)
        .function(&["tau2"], exp_decay)
        .partial_deriv("tau2", exp_decay_dtau)
        .invariant_function(|x|DVector::from_element(x.len(),1.))
        .build()
        .expect("double exponential model builder should produce a valid model");
    // 2. Cast the fitting problem as a nonlinear least squares minimization problem
    let problem = LevMarProblemBuilder::new()
        .model(&model)
        .x(x)
        .y(y)
        .initial_guess(init_values)
        .build()
        .expect("Building valid problem should not panic");
    // 3. Solve the fitting problem
    let (solved_problem, report) = LevMarSolver::new().minimize(problem);
    assert!(report.termination.was_successful());
    // 4. obtain the nonlinear parameters after fitting
    // let alpha = solved_problem.params();
    let alpha = solved_problem.params();
    let c = solved_problem.linear_coefficients().unwrap();
    // println!("nfev={}, f(x)={}", report.number_of_evaluations, report.objective_function);
    // let yf = model.eval(&x, alpha.as_slice()).unwrap()*c.clone();
    let p = DVector::<f64>::from_vec(vec![alpha[0], alpha[1], c[0], c[1], c[2]]);
    (report.objective_function, p)
}
fn main() {

}