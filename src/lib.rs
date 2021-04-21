pub mod evolutionary_optimizer;
pub mod levenberg_marquardt;
pub mod varpro;
pub mod models;

extern crate oxigen;
extern crate rand;
extern crate pyo3;
extern crate numpy;

// use std::iter::FromIterator;
use std::collections::HashMap;

use nalgebra::{DVector};
use ndarray::{Array, s};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python, Py, PyAny, IntoPy};
use varpro::NExponentialProblemVarpro;

impl IntoPy<Py<PyAny>> for NExponentialProblemVarpro {
    fn into_py(self, py: pyo3::Python<'_>) -> Py<PyAny> { 
        let mut result: HashMap<&str,&PyArrayDyn<f64>> = HashMap::new();
        // let arr = &Array::from_iter(m.iter().map(|&x|x));
        // let arr = &Array::from(*m.data.as_slice().clone());
        // Array::linspace(42.2, 43.5, 20).into_dyn().into_pyarray(py);
        let mut into_result = |k, x: DVector<f64>| {
            let mut arr = Array::zeros(x.len());
            for i in 0..x.len() {
                arr[i] = x[i];
            }
            result.insert(k, arr.into_dyn().into_pyarray(py));
        };

        into_result("y_est", self.y_est);
        into_result("wresid", self.wresid.unwrap());
        into_result("alpha", self.alpha);
        into_result("c", self.c);
        // result.insert("y_est", );
        result.into_py(py)
    }
}

#[pymodule]
fn quifi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    #[pyfn(m, "say_hello")]
    fn say_hello_py() -> String {
        String::from("Hi, im quifi.")
    }
    #[pyfn(m, "optional")]
    fn optional_py(x: Option<f64>) -> String {
        match x {
            Some(v) => format!("Your reciprocal number is {}.", v.recip()),
            None => String::from("No number supplied."),
        }
    }

    #[pyfn(m, "find_solution")]
    fn find_solution_py<'py>(
        py: Python<'py>,
        xdata: PyReadonlyArrayDyn<f64>,
        ydata: PyReadonlyArrayDyn<f64>,
    // ) -> &'py PyArrayDyn<f64> {
    ) -> Py<PyAny> {
        // arr.slice(s![0,..]);
        let x = xdata.as_array();
        let y = ydata.as_array();
        assert!(x.dim() == y.dim(), "X and Y not the same dimension.");
        assert!(x.ndim() == 1, "Wrong input dimension.");
        let y: Vec<f64> = y.slice(s![..]).to_vec();
        let x: Vec<f64> = x.slice(s![..]).to_vec();
        let result = evolutionary_optimizer::find_solution(&x, &y, None, None, None, None);
        // let x: Vec<f64> = (0..y.len()).map(|x| (x as f64)/146.0).collect();
        // let x DVector::<f64>::from_iter((0..146).map(|x| (x as f64)/146.0));
        // println!("Final  residuals: {}", result.wresid.unwrap().norm());
        // println!("Final parameters: {}", result.alpha);
        // Array::linspace(42.2, 43.5, 20).into_dyn().into_pyarray(py)
        result.into_py(py)
    }
    Ok(())
}