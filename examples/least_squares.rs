#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_doc_comments)]
extern crate levenberg_marquardt;
use levenberg_marquardt::{LeastSquaresProblem,LevenbergMarquardt,differentiate_numerically};
extern crate nalgebra;
use nalgebra::{DMatrix, Dynamic,DVector};
use nalgebra::storage::Owned;
extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2,s};
use ndarray_npy::ReadNpyExt;
mod plot;
use plot::plot_results;

#[derive(Clone, PartialEq, Debug)]
pub struct NExponentialProblem {
    // holds current value of the n parameters
    pub x: DVector<f64>,
    pub y: DVector<f64>,
    pub p: DVector<f64>, // parameters
}

#[derive(Clone, PartialEq, Debug)]
pub struct NExponentialProblemVarpro {
    // holds current value of the n parameters
    pub x: DVector<f64>,
    pub y: DVector<f64>,
    pub w: DVector<f64>, // weights
    pub p: DVector<f64>, // parameters
    pub c: DVector<f64>, // linear coefficients
}

fn read_example(input:&str) -> Result<ndarray::Array2<f64>, ndarray_npy::ReadNpyError> {
    let reader = std::fs::File::open(input)?;
    let arr = Array2::<f64>::read_npy(reader)?;
    Ok(arr)
}

fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x| (-x*tau).exp())
}

fn exp_decay_dtau(x: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  x.map(|x| -x*(-x * tau).exp())
}

fn linear_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) -> DVector<f64> {
    // solves the problem min_a ||x*a-y||^2
    // let m = (x.transpose()*x.clone()).pseudo_inverse(1e-9).expect("Could not invert input");
    // m*x.transpose()*y

    // let svd = nalgebra::linalg:SVD::new(x, true, true);
    let svd = x.clone().svd(true, true);
    svd.pseudo_inverse(1e-9).unwrap()*y
}
// trait VarproSolver: LeastSquaresProblem<f64, Dynamic, Dynamic> {}
// We implement a trait for every problem we want to solve
impl LeastSquaresProblem<f64, Dynamic, Dynamic> for NExponentialProblemVarpro {
    type ParameterStorage = Owned<f64, Dynamic>;
    type ResidualStorage = Owned<f64, Dynamic>;
    type JacobianStorage = Owned<f64, Dynamic, Dynamic>;
     
    fn set_params(&mut self, p: &DVector<f64>) {
        std::assert!(p.len() == self.c.len() - 1, "Parameter count p={}, c={} is invalid", p.len(), self.c.len());
        self.p.copy_from(p);
        // do common calculations for residuals and the Jacobian here
    }
     
    fn params(&self) -> DVector<f64> { self.p.clone() }

    fn residuals(&self) -> Option<DVector<f64>> {
        let mut phiv = vec![DVector::from_element(self.x.len(), 1.0)];
        for i in 0..self.p.len() {
            phiv.push(exp_decay(&self.x, self.p[i])); 
        }
        // let ma = (0..self.p.len()).map(|i| exp_decay(&self.x, self.p[i])).collect::<Vec<DVector<f64>>>(); 
        let phim =  DMatrix::<f64>::from_columns(phiv.as_slice());
        // let c = linear_least_squares(&phim, &self.y);
        let svd = phim.clone().svd(true, true);
        let c = svd.pseudo_inverse(1e-9).unwrap()*self.y.clone();
        let f = phim*c;
        Some(f - self.y.clone())
    }
    // fn jacobian(&self) -> Option<DMatrix<f64>> {
    //     differentiate_numerically(&mut self.clone())
    // }
     
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let mut dv = vec![DVector::from_element(self.x.len(), 1.0)];
        for i in 0..self.p.len() {
            dv.push(exp_decay_dtau(&self.x, self.p[i])); 
        }
        // let ma = (0..self.p.len()).map(|i| exp_decay(&self.x, self.p[i])).collect::<Vec<DVector<f64>>>(); 
        let dk =  DMatrix::<f64>::from_columns(dv.as_slice());

        let mut phiv = vec![DVector::from_element(self.x.len(), 1.0)];
        for i in 0..self.p.len() {
            phiv.push(exp_decay(&self.x, self.p[i])); 
        }
        let phim =  DMatrix::<f64>::from_columns(dv.as_slice());

        let svd = phim.clone().svd(true, true);

        let u = svd.u.clone().unwrap();
        let v_t = svd.v_t.clone().unwrap();
        let s = DMatrix::<f64>::from_diagonal(&svd.singular_values.map(|x| x.recip()));
        let c = svd.pseudo_inverse(1e-9).unwrap()*self.y.clone();
        let rw = self.y.clone() - phim*c.clone();
        // let Dk = dm.column_iter().map(|col| );
        let mut dkc = dk.clone();
        for (i, mut col) in dkc.column_iter_mut().enumerate() {
            col *= c[i];
        }
        let mut dkr = dk.clone();
        for (i, mut col) in dkr.column_iter_mut().enumerate() {
            for j in 0..col.len() {
                col[j] *= rw[j];
            }
        }
        // let Dkc = dm*c*self.y.clone();
        // let DkTr = dm*self.y.clone();
        // (rows: a, columns: b) * (rows: b, columns: c) -> (rows:a, columns: c)
        // vectors of dimension n have (rows: 1, columns: n)
        let a = dkc.clone() - u.clone()*(u.transpose()*dkc);
        // let b = u*(s*(v_t*(dk.transpose()*rw)));
        let mut b = u*s*v_t;
        for (i, mut col) in b.column_iter_mut().enumerate() {
            for j in 0..col.len() {
                col[j] *= dkr[(j,i)]*rw[j];
            }
        }
        println!("");
        // let ma = (0..self.p.len()).map(|i| exp_decay(&self.x, self.p[i])).collect::<Vec<DVector<f64>>>(); 
        // let phim =  DMatrix::<f64>::from_columns(phiv.as_slice()).transpose();
        Some(-(a.remove_column(0) + b.remove_column(0)))
    }
}

// We implement a trait for every problem we want to solve
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
fn main() {
    let arr = read_example("data/tesdata.npy").unwrap();
    println!("Local optimizer test.");
    let yv = arr.slice(s![61, ..146]).to_vec();
    let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
    /**
     * LEVENBERG MARQUARDT
     */
    let mut problem = NExponentialProblem {
        p: DVector::<f64>::from_vec(vec![1., 1., 1., 1., 1.]),
        x: DVector::<f64>::from_vec(xv.clone()),
        y: DVector::<f64>::from_vec(yv.clone()),
    };

    let jacobian_numerical = differentiate_numerically(&mut problem).unwrap();
    let jacobian_trait = problem.jacobian().unwrap();
    // approx::assert_relative_eq!(jacobian_numerical, jacobian_trait, epsilon = 1e-13, );
    println!("Relative norm of difference {}", 
        (jacobian_numerical.clone() - jacobian_trait.clone()).norm()
        /jacobian_numerical.norm());
    // println!("{:.6e}", (jacobian_trait));
    // println!("{:.6e}", (jacobian_numerical));
    let (result, report) = LevenbergMarquardt::new().minimize(problem.clone());
    // assert!(report.objective_function.abs() < 1e-10);
    println!("Final Residuals {}", report.objective_function.abs());
    println!("Optimal parameters {}", result.p);
    plot_results(&result.x, &result.y, &(result.residuals().unwrap()+result.y.clone()), "data/levenberg_marquardt.png").unwrap();
    /**
     * LINEAR REGRESSION
     */
    let m = DMatrix::from_columns(&[DVector::from_element(result.x.len(), 1.0), result.x.clone()]);
    let a = linear_least_squares(&m, &result.y);
    println!("Linear regression {}", a);
    plot_results(&result.x, &result.y, &(m*a), "data/linear_regression.png").unwrap();

    /**
     * VARPRO WITH LM
     */
    let mut problem = NExponentialProblemVarpro {
        p: DVector::<f64>::from_vec(vec![1., 1.]),
        w: DVector::<f64>::from_element(xv.len(), 1.0),
        x: DVector::<f64>::from_vec(xv),
        y: DVector::<f64>::from_vec(yv),
        c: DVector::<f64>::zeros(3),
    };
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
    // println!("Numerical {}", jacobian_numerical.clone());
    // println!("Analytical{}", jacobian_trait.clone());
}