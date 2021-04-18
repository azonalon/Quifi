#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_doc_comments)]
extern crate criterion;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

extern crate levenberg_marquardt;
use levenberg_marquardt::{LeastSquaresProblem,LevenbergMarquardt,differentiate_numerically};
extern crate nalgebra;
use nalgebra::{DMatrix, Dynamic,DVector};
use nalgebra::storage::Owned;
extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2};
use ndarray_npy::ReadNpyExt;
// mod plot;
// mod jacobian;
// use plot::plot_results;

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
    pub alpha: DVector<f64>, // nonlinear parameter count
    pub n: usize, // number of linear parameters
    pub n1: usize, // number of basis functions
    U: DMatrix<f64>, 
    V: DMatrix<f64>, // phi = U*s*V^T
    s: DVector<f64>, // singular values
    pub c: DVector<f64>, // linear parameters
    pub m: usize, // length of sample data x and y
    pub p: usize, // nonzero columns of partial derivatives
    pub q: usize, // number of nonlinear parameters (alpha.len())
    pub wresid: Option<DVector<f64>>, // weighted residuals
    pub phi: DMatrix<f64>, // nonlinear functions values
    pub d_phi: DMatrix<f64>, // non-zero nonlinear function derivatives
    pub ind: DMatrix<usize>, // index into d_phi
    pub rank: usize, // rank of phi (singular values > tol)
    pub y_est: DVector<f64>, // output of the model function
}
impl NExponentialProblemVarpro {
    pub fn new(
        x: DVector<f64>, y: DVector<f64>, w: DVector<f64>, // weights
        alpha: DVector<f64>, // nonlinear parameter count
        n: usize, // number of linear parameters
        n1: usize, // number of basis functions including an optional constant f=1
    ) -> Self {
        let mv = DMatrix::<f64>::zeros(0,0);
        let vv = DVector::<f64>::zeros(0);
        let mut s = Self {
            x: x, 
            y: y,
            w: w, // weights
            alpha: alpha.clone(), // nonlinear parameter count
            n: n, // number of linear parameters
            n1: n1, // number of basis functions
            U: mv.clone(), 
            V: mv.clone(), // phi = U*s*V^T
            s: vv.clone(), // singular values
            c: vv.clone(), // linear parameters
            m: 0, // length of sample data x and y
            p: 0, // nonzero columns of partial derivatives
            q: 0, // number of nonlinear parameters (alpha.len())
            wresid: None, // weighted residuals
            phi: mv.clone(), // nonlinear functions values
            d_phi: mv, // non-zero nonlinear function derivatives
            ind: DMatrix::<usize>::from_row_slice(2, 3,
                &[0, 1, 2,
                  0, 1, 2] // todo, compute from parameter count, this is a 2 x p
            ), // index into d_phi
            rank: 0, // rank of phi (singular values > tol)
            y_est: vv,
        };
        s.set_params(&alpha);
        s
    }
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
     
    fn set_params(&mut self, alpha: &DVector<f64>) {

        self.q = self.alpha.len();
        self.p = self.ind.ncols();
        self.m = self.x.len();
        self.alpha.copy_from(alpha);

        self.d_phi =  DMatrix::<f64>::zeros(self.x.len(), self.q);
        for i in 0..self.q {
            self.d_phi.set_column(i, &exp_decay_dtau(&self.x, self.alpha[i])); 
        }


        assert!(self.q + 1 == self.n);
        assert!(self.d_phi.shape() == (self.m, self.q));

        self.phi =  DMatrix::<f64>::zeros(self.x.len(), self.n);
        for i in 0..self.q { // TODO: q is not always the right variable here
            self.phi.set_column(i, &exp_decay(&self.x, self.alpha[i])); 
        }
        self.phi.set_column(self.alpha.len(), &DVector::from_element(self.x.len(), 1.0));

        assert!(self.phi.shape() == (self.m, self.n1));

        let tol = (self.m as f64) * 1e-12;
        let some_svd = self.phi.clone().try_svd(true, true, tol, 100);
        if some_svd.is_none() {
            self.wresid = None;
            return;
        }
        let svd = some_svd.unwrap();
        self.rank = svd.rank(tol); 
        // NOTE: matlab 'svd' creates blown matrices. newer 
        // algorithms reduce the redundant columns in svd
        self.U = svd.u.clone().unwrap().columns(0, self.rank).into();
        self.V = svd.v_t.clone().unwrap().transpose().columns(0, self.rank).into();
        self.s = svd.singular_values.clone().rows(0, self.rank).into();


        let mut yuse = self.y.clone();
        if self.n < self.n1 {
            panic!("This case is not actually implemented");
            yuse  +=  - self.phi.column(self.n1 - 1);// % extra function Phi(:,n+1)
        }
        if self.rank < self.n {
            // println!("Warning, input functions evulations show degeneracy.");
            // println!("This leads to numerical instability.");
            // println!("Maybe try to adjust initial parameters to not be equal.");
        }
        // let temp  = self.U.transpose() * (self.w.zip_map(&yuse, |x, y| x*y));    
        // self.c = self.V.clone() * (temp.zip_map(&self.s, |x,y| x/y));
        // self.c = self.V.clone()* (
        //     (self.U.transpose()*self.w.component_mul(&yuse))
        //     .component_div(&self.s)
        // );
        self.c = svd.pseudo_inverse(tol).unwrap() * self.w.component_mul(&yuse);

        self.y_est = self.phi.columns(0, self.n) * self.c.clone();
        self.wresid = Some((yuse - self.y_est.clone()).component_mul(&self.w));
        if self.n < self.n1 {
            self.y_est += self.phi.column(self.n1 - 1);
        }
    }
     
    fn params(&self) -> DVector<f64> { 
        // println!("params called.");
        self.alpha.clone() 
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        // println!("residuals called.");
        self.wresid.clone()
    }
    // fn jacobian(&self) -> Option<DMatrix<f64>> {
    //     differentiate_numerically(&mut self.clone())
    // }
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        // println!("jacobian called.");
        // Some(jacobian::form_jacobian(&self.x, &self.y, &self.p, &self.w,
        // &self.phi, &self.d_phi, &self.ind, &self.U, &self.V, &self.s, self.rank, 2, 3)) 
        if self.wresid.is_none() {
            return None;
        }
        let wresid = self.wresid.clone().unwrap();
        assert!(self.d_phi.shape() == (self.m, self.q), "Wrong shape for dphi {:?}", self.d_phi.shape());
        assert!(self.phi.shape()  == (self.m, self.n), "Wrong shape for phi {:?}", self.phi.shape());
        assert!(wresid.shape()  == (self.m, 1), "Wrong shape for wresd {:?}", wresid.shape());
        let w_dphi_r: nalgebra::DVector::<f64> = self.d_phi.transpose()*wresid.clone();
        let mut w_d_phi  = self.d_phi.clone();
        for (i, mut row) in w_d_phi.row_iter_mut().enumerate() {
            row *= self.w[i];
        }
        assert!(w_dphi_r.shape()  == (self.q,1), "w_dphi_r should be a vector");
        let mut t_2 = nalgebra::DMatrix::<f64>::zeros(self.n1, self.q);
        let mut jac_1 = nalgebra::DMatrix::<f64>::zeros(self.m, self.q);
        // let Jac2 = Matrix::zeros(m, p);
        let mut ctemp = self.c.clone();
        if self.n1 > self.n {
            let l = ctemp.len();
            ctemp = ctemp.insert_row(l, 1.0);
        }
        for j in 0..self.q {                          //  % for each nonlinear parameter alpha(j)
            // range = find(Ind(2,:)==j);      //  % columns of WdPhi relevant to alpha(j)
            // indrows = Ind(1,range);         //  % relevant rows of ctemp
            // IND: 2 x p, WdPHI m x p, JAC m x p, m: x.len(), p : parameter count
            // T2: n1 x q
            let mut range = vec![];
            self.ind.row(1).iter().enumerate().for_each(|(k, x)| if *x==j {range.push(k)});
            let indrows = self.ind.row(0).select_columns(range.iter()).transpose();
            let cola = w_d_phi.select_columns(range.iter());
            let col: nalgebra::DVector<f64> = 
                cola * ctemp.select_rows(indrows.as_slice().iter());
            jac_1.set_column(j, &col);      
            for (&i, &k) in indrows.iter().zip(range.iter()) {
                t_2[(i,j)] = w_dphi_r[k];
            }
            // range = find(Ind(2,:)==j);        % columns of WdPhi relevant to alpha(j)
            // indrows = Ind(1,range);           % relevant rows of ctemp
            // Jac1(:,j) = WdPhi(:,range) * ctemp(indrows);      
            // T2(indrows,j) = WdPhi_r(range);
        }


        // assert!(self.m-self.rank-1 > 0, "some index error");
        // Jac1 = U(:,myrank+1:m) * (U(:,myrank+1:m)' * Jac1); 
        // NOTE: matlab svd contains redundant columns, such the next line
        let u_s = DMatrix::<f64>::identity(self.m, self.m) - self.U.clone() * self.U.transpose();
        // jac_1 = u_s.clone() * (u_s.transpose() * jac_1);
        jac_1 = u_s * jac_1;


        // T2 = diag(1 ./ s(1:myrank)) * (V(:,1:myrank)' * T2(1:n,:));
        t_2 = nalgebra::DMatrix::<f64>::from_diagonal(&self.s.map(|x| x.recip())) * 
                    (self.V.transpose() * t_2.rows(0, self.n));
        // Jac2 = U(:,1:myrank) * T2;
        let jac_2 = self.U.clone() * t_2;

        // Jacobian = -(Jac1 + Jac2);
        Some(-(jac_1 + jac_2))
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
    let yv = arr.slice(ndarray::s![61, ..146]).to_vec();
    let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
    /**
     * LEVENBERG MARQUARDT
     */
    if false {
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
        // plot_results(&result.x, &result.y, &(result.residuals().unwrap()+result.y.clone()), "data/levenberg_marquardt.png").unwrap();
        /**
         * LINEAR REGRESSION
         */
        let m = DMatrix::from_columns(&[DVector::from_element(result.x.len(), 1.0), result.x.clone()]);
        let a = linear_least_squares(&m, &result.y);
        println!("Linear regression {}", a);
        // plot_results(&result.x, &result.y, &(m*a), "data/linear_regression.png").unwrap();

        // let problem = NExponentialProblemVarpro {
        //     alpha: DVector::<f64>::from_vec(vec![1., 1.]),
        //     n:3, n1:2,
        //     w: DVector::<f64>::from_element(xv.len(), 1.0),
        //     x: DVector::<f64>::from_vec(xv),
        //     y: DVector::<f64>::from_vec(yv),
        //     // ..Default::default()
        // };
    }
    /*
    * VARPRO WITH LM
    */
    let mut problem = NExponentialProblemVarpro::new(
        DVector::<f64>::from_vec(xv.clone()),
        DVector::<f64>::from_vec(yv),
        DVector::<f64>::from_element(xv.len(), 1.0),
        DVector::<f64>::from_vec(vec![1., -1., 2.2]),
        4, // n, number of linear parameters
        4, // n1, basis functions (including constant term)
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

    println!("Final Residuals {}", report.objective_function.abs());
    println!("Optimal parameters {}", result.alpha);
    // plot_results(&result.x, &result.y, &result.y_est, "data/varpro.png").unwrap();
}