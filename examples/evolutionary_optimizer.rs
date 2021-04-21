extern crate ndarray;
extern crate nalgebra;
// extern crate quick_fitting;
use ndarray::s;
// extern crate quick_fitting;
use quifi::evolutionary_optimizer::find_solution;
mod npy_io;


fn main() {
    let arr = npy_io::read_npy("examples/exponential_data.npy").unwrap();
    // arr.slice(s![0,..]);
    let y: Vec<f64> = arr.slice(s![0,..]).to_vec();
    let x: Vec<f64> = (0..y.len()).map(|x| (x as f64)/146.0).collect();
    // let x DVector::<f64>::from_iter((0..146).map(|x| (x as f64)/146.0));
    let result = find_solution(&x, &y, None, None, None, None);
    println!("Final  residuals: {}", result.wresid.unwrap().norm());
    println!("Final parameters: {}", result.alpha);
}