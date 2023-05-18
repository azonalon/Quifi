extern crate ndarray;
extern crate nalgebra;
// extern crate quick_fitting;
// extern crate quick_fitting;
use quifi::evolutionary_optimizer::find_solution;
mod npy_io;
use rand::{thread_rng};
use rand_distr::{Distribution};


fn main() {
    let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();
    let x: Vec::<f64> = ((0..300).map(|x| (x as f64)/300.0)).collect();
    let y = x.iter().map(|x| 0.001*dist.sample(&mut rng) + 3.0*(-x*2.0).exp() + 5.0*(-x*3.0).exp()).collect();

    let (result, exec) = find_solution(&x, &y, None, None, None, None);
    println!("Final residuals: {}", result.wresid.unwrap().norm());
    println!("Final parameters: {}", result.alpha);
    println!("History: {:#?}", exec.individual_history.unwrap());
}