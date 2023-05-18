#![allow(unused_imports)]
pub mod evolutionary_optimizer;
pub mod levenberg_marquardt;
pub mod varpro;
pub mod models;
pub mod regression;

extern crate oxigen;
extern crate rand;
extern crate pyo3;
extern crate numpy;

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector, Dynamic, Vector, iter::MatrixIter, storage::Storage};
use ndarray::{Array, Array2, ArrayView2, Ix2, s, IxDyn};
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, ToPyArray, PyArray};
use pyo3::{
    prelude::{
        pymodule, PyModule, PyResult, Python, Py, PyAny, IntoPy, IntoPyPointer, ToPyObject
    }, 
    types::{
        IntoPyDict, PyDict, PyFloat, PyTuple
    }
};
use varpro::LevenbergMarquardtVarproProblem;
use regression::Statistics;
use evolutionary_optimizer::*;
use oxigen::*;


impl IntoPy<Py<PyAny>> for LevenbergMarquardtVarproProblem {
    fn into_py<'py>(self, py: pyo3::prelude::Python<'py>) -> Py<PyAny> { 
        let result = PyDict::new(py);
        let regression_data = self.statistics();

        let insert_result_vector = |k, x: DVector<f64>| {
            let mut arr = Array::zeros(x.len());
            for i in 0..x.len() {
                arr[i] = x[i];
            }
            result.set_item(k, arr.into_pyarray(py)).unwrap();
        };
        let err = regression_data.standard_error;
        let sigma = regression_data.sigma.to_object(py);
        let rms   = regression_data.rms  .to_object(py);
        let rank   = self.rank           .to_object(py);

        insert_result_vector("y_est",  self.y_est);
        insert_result_vector("wresid", self.wresid.unwrap());
        insert_result_vector("p_opt",  self.alpha);
        insert_result_vector("c_opt",  self.c);
        insert_result_vector("p_err",  err.rows(self.n, self.q).into());
        insert_result_vector("c_err",  err.rows(0,      self.n).into());

        let insert_result_matrix = |k, x: DMatrix<f64>| {
            let view = ArrayView2::<f64>::from_shape(
                (x.nrows(), x.ncols()), 
                x.data.as_slice()
            ).ok().unwrap().into_dyn();
            result.set_item(k, view.to_pyarray(py)).unwrap();
        };

        insert_result_matrix("c_p_cov", regression_data.covariance);
        insert_result_matrix("c_p_cor", regression_data.correlation);

        result.set_item("sigma", sigma).unwrap();
        result.set_item("rms"  , rms).unwrap();
        result.set_item("rank" , rank).unwrap();

        result.into_py(py)
    }
}

#[pymodule]
fn quifi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    #[pyfn(m)]
    #[pyo3(name="say_hello")]
    fn say_hello_py() -> String {
        String::from("Hi, im the new quifi.")
    }

    #[pyfn(m, kwargs="**")]
    #[pyo3(name="optional")]
    fn optional_py(x: Option<f64>, kwargs: Option<&PyDict>) -> String {
        match kwargs {
            Some(v) => println!("kwargs supplied: {:#?}", v),
            None => println!("No kwargs supplied."),
        };
        match x {
            Some(v) => format!("Your reciprocal number is {}.", v.recip()),
            None => String::from("No number supplied."),
        }
    }

    #[pyfn(m, kwargs="**")]
    #[pyo3(name="find_solution")]
    fn find_solution_py<'py>(
        py: Python<'py>,
        xdata: PyReadonlyArrayDyn<f64>,
        ydata: PyReadonlyArrayDyn<f64>,
        kwargs: Option<&PyDict>
    ) -> Py<PyAny> {
        let x = xdata.as_array();
        let y = ydata.as_array();
        assert!(x.dim() == y.dim(), "X and Y not the same dimension.");
        assert!(x.ndim() == 1, "Wrong input dimension.");
        let y: Vec<f64> = y.slice(s![..]).to_vec();
        let x: Vec<f64> = x.slice(s![..]).to_vec();
        let ps = Points {
            x: DVector::from(x.clone()), 
            y: DVector::from(y.clone())
        };

        let mut n_parameters: i64 = 2;
        match kwargs {
            Some(d) => { 
                match d.get_item("n_basefunctions") {
                    Some(v) => n_parameters = v.extract().unwrap(),
                    None => {},
                }
                match d.get_item("routine") {
                    Some(v) => n_parameters = v.extract().unwrap(),
                    None => {},
                }
            }
            None => {},
        }

        let log2 = ((n_parameters as f64) * 4_f64).log2().ceil();
        // let progress_log = File::create("data/progress2.csv").expect("Error creating progress log file");
        // let population_log = File::create("data/population.txt").expect("Error creating population log file");
        let population_size = 2_i32.pow(log2 as u32) as usize;
        let nt = NTournaments(population_size/2);
        let mut exec = GeneticExecution::<f64, NonlinearParameters>::new()
            .population_size(population_size)
            .genotype_size(n_parameters)
            .fitness_arguments(ps)
            .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
                start: (n_parameters as f64) / (8_f64 + 2_f64 * log2) / 100_f64,
                bound: 0.10,
                coefficient: -0.0002,
            })))
            // .stop_criterion(Box::new(StopCriteria::Generation(50)))
            .stop_criterion(Box::new(StopCriteria::SolutionsFoundOrGeneration(1, 20)))
            .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
                start: log2 - 2_f64,
                bound: log2 / 1.5,
                coefficient: -0.0005,
            })))
            .select_function(Box::new(SelectionFunctions::Tournaments(nt)))
            .age_function(Box::new(AgeFunctions::Quadratic(
                AgeThreshold(50),
                AgeSlope(1_f64),
            )))
            .generate_history(true)
            ;
            // .progress_log(20, progress_log)
            // .population_log(2000, population_log)
        let (solutions, _generation, _progress) = exec.run();
        let population = exec.population.clone();
        assert!(population.len() > 0, "No Population ?");
        let mut s = solutions.iter().min_by(
            |x, y| 
            x.result.clone().unwrap().wresid.unwrap().norm().partial_cmp(
            &y.result.clone().unwrap().wresid.unwrap().norm() ).unwrap()
        );
        if s.is_none() {
            let it = population.iter().filter(|x| x.ind.result.clone().unwrap().wresid.is_some());
            s = Some(&it.min_by(
                |x, y| 
                x.ind.result.clone().unwrap().wresid.unwrap().norm().partial_cmp(
                &y.ind.result.clone().unwrap().wresid.unwrap().norm() ).unwrap()
            ).unwrap().ind);
        }
        if s.is_none() {
            println!("wow.. {} {}", solutions.len(), population.len());
            // s = Some(&population.iter().next().unwrap().ind.clone());
        }
        let sol = s.unwrap();
        let result = sol.result.clone().unwrap();
        // (result, exec)
            // let (result, exec) = evolutionary_optimizer::find_solution(&x, &y, None, None, None, None);
        (result.into_py(py), exec.individual_history.unwrap().into_py(py)).into_py(py)
    }
    Ok(())
}