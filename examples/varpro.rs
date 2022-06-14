// extern crate quick_fitting;
extern crate rand;
use nalgebra::{DVector};
use rand::{thread_rng};

use quifi::varpro::NExponentialProblemVarpro;
use rand_distr::Distribution;
mod plot;
extern crate rand_distr;
use quifi::regression::Statistics;
use levenberg_marquardt::{differentiate_numerically, 
                          LevenbergMarquardt, 
                          LeastSquaresProblem};
fn main() {
    /*
    * VARPRO WITH LM
    */
    // const INC: u64 = 11634580027462260723;
    let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let x = DVector::<f64>::from_iterator(300, (0..300).map(|x| (x as f64)/300.0));
    let y = x.map(|x| 0.01*dist.sample(&mut rng) + -2.0*(-x*2.0).exp() + 1.0*(-x*5.0).exp()+2.0);
    

    let mut problem = NExponentialProblemVarpro::new(
        x.clone(), y.clone(),
        DVector::<f64>::from_element(x.len(), 1.0),
        DVector::<f64>::from_vec(vec![1.0, 2.0]),
        3, // n, number of linear parameters
        3, // n1, basis functions (including constant term)
    );
    // let (result, report) = LevenbergMarquardt::new().minimize(problem.clone());
    // println!("Final Residuals {}", report.objective_function.abs());
    // println!("Optimal parameters {}", result.p);
    // plot_results(&result.x, &result.y, &(result.residuals().unwrap()+result.y.clone()), "data/varpro.png").unwrap();

    let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    // approx::assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13, );
    println!("Relative norm of difference {}", 
        (jacobian_numerical.clone() - jacobian_trait.clone()).norm()
        /jacobian_numerical.norm());

    // Criterion::default()
    //     .sample_size(40)
    //     .bench_function("fib 20", |b| b.iter(|| 
    //         LevenbergMarquardt::new().minimize(problem.clone())
    // ));
    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    // assert!(report.objective_function.abs() < 1e-10);
    let stats = result.statistics();

    println!("Final Residuals {}", report.objective_function.abs());
    println!("Optimal parameters {}", result.alpha);
    println!("Covariance {:.3}", stats.covariance);
    println!("Correlation {:.3}", stats.correlation);
    println!("Parameter error {:.3}", stats.standard_error);
    println!("t ratio {:.3}", stats.t_ratio);
    plot::plot_results(&result.x, &result.y, &result.y_est, "data/varpro.png").unwrap();
}