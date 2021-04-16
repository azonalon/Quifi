#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2,s};
use ndarray_npy::ReadNpyExt;
use nalgebra::{DVector,DMatrix};
use varpro::prelude::*;
use varpro::solvers::levmar::{LevMarProblemBuilder, LevMarSolver};
use levenberg_marquardt::{differentiate_numerically};


fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x|(-x*tau).exp())
}

fn exp_decay_dtau(tvec: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  tvec.map(|t| t*((-t * tau).exp()))
}



mod plot;
use plot::plot_results;

fn read_example(input:&str) -> Result<ndarray::Array2<f64>, ndarray_npy::ReadNpyError> {
    let reader = std::fs::File::open(input)?;
    let arr = Array2::<f64>::read_npy(reader)?;
    Ok(arr)
}

fn main() {
    // let arr = read_example("data/tesdata.npy").unwrap();
    // let n = arr.shape()[0];
    // println!("@len={}\n", );
    // let mut v_error = DVector::<f64>::from_element(n, 0.0);
    // for k in 0..n {
    //     let init_values = vec![5.170, 5.668, 7.915];
    //     let yv = arr.slice(s![k, ..]).to_vec();
    //     let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
    //     let x = nalgebra::DVector::<f64>::from_vec(xv);
    //     let y = nalgebra::DVector::<f64>::from_vec(yv);
    //     // let (s, p) = fit_biexponential(x, y, &[1.0, 2.0]);
    //     let (s, p) = fit_triexponential(x, y, init_values.as_slice());
    //     v_error[k] = s;
    // }
    // plot_results(DVector::<f64>::from_iterator(n, (0..n).map(|x| x as f64)), v_error.clone(), v_error.clone(), "data/f_error_local.png").unwrap();

    let arr = read_example("data/tesdata.npy").unwrap();
    println!("Local optimizer test.");
    // let init_values = vec![5.170, 5.668, 7.915];
    let init_values = vec![10.384829253417529, 19.184026086321197, 14.2594866037982];
    // let init_values = vec![1.0, 1.0, 1.0];
    // let yv = arr.slice(s![61, ..]).to_vec();
    // let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
    let yv = arr.slice(s![61, ..10]).to_vec();
    let xv = ndarray::Array::linspace(0., 10., 10).to_vec();
    let x = nalgebra::DVector::<f64>::from_vec(xv);
    let y = nalgebra::DVector::<f64>::from_vec(yv);
    // let (s, p) = fit_biexponential(x, y, &[1.0, 2.0]);
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
    let mut problem = LevMarProblemBuilder::new()
        .model(&model)
        .x(x)
        .y(y)
        // .epsilon(0.00000)
        .initial_guess(init_values.as_slice())
        .build()
        .expect("Building valid problem should not panic");
    // 3. Solve the fitting problem
    println!("minimize...");
    let user_tol = 1e-5_f64;
    // 4. obtain the nonlinear parameters after fitting
    // let alpha = solved_problem.params();
    // println!("nfev={}, f(x)={}", report.number_of_evaluations, report.objective_function);
    // let yf = model.eval(&x, alpha.as_slice()).unwrap()*c.clone();
    // vec![alpha[0], alpha[1], c[0], c[1], c[2]]
    // let v = Vec::<f64>::new();
    let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    // approx::assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13, );
    println!("Norm of difference{}", (jacobian_numerical.clone() - jacobian_trait.clone()).norm()/jacobian_numerical.norm());
    println!("{:.6e}", (jacobian_trait));
    println!("{:.6e}", (jacobian_numerical));
    panic!();
    let (solved_problem, report) = LevMarSolver::new()
        .with_ftol(user_tol)
        .with_gtol(user_tol)
        // .with_patience(100)
        // .with_stepbound(1.0)
        .minimize(problem.clone());
    println!("minimized...");
    assert!(report.termination.was_successful());
    // let alpha = solved_problem.params();
    // let c = solved_problem.linear_coefficients().unwrap();
    // let p = DVector::<f64>::from_vec(vec![alpha[0], alpha[1], alpha[2], c[0], c[1], c[2], c[3]]);
}