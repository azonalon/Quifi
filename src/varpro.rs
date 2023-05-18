use levenberg_marquardt::{LeastSquaresProblem};
use nalgebra::{DMatrix, Dynamic,DVector};
use nalgebra::storage::Owned;
use crate::models::{exp_decay,exp_decay_dtau};
// use pyo3::{IntoPy, types::PyFloat, PyAny, Py};

pub struct VarproModel {
    
}
#[derive(Clone, PartialEq, Debug)]
pub struct LevenbergMarquardtVarproProblem {
    // holds current value of the n parameters
    pub x: DVector<f64>, 
    pub y: DVector<f64>,
    pub w: DVector<f64>, // weights
    pub alpha: DVector<f64>, // nonlinear parameter count
    pub n: usize, // number of linear parameters
    pub n1: usize, // number of basis functions
    u: DMatrix<f64>, 
    v: DMatrix<f64>, // phi = U*s*V^T
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


impl LevenbergMarquardtVarproProblem {
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
            u: mv.clone(), 
            v: mv.clone(), // phi = U*s*V^T
            s: vv.clone(), // singular values
            c: vv.clone(), // linear parameters
            m: 0, // length of sample data x and y
            p: 0, // nonzero columns of partial derivatives
            q: 0, // number of nonlinear parameters (alpha.len())
            wresid: None, // weighted residuals
            phi: mv.clone(), // nonlinear functions values
            d_phi: mv, // non-zero nonlinear function derivatives
            ind: DMatrix::<usize>::from_row_slice(2, 6,
                &[0, 1, 2, 3, 4, 5,
                  0, 1, 2, 3, 4, 5] // todo, compute from parameter count, this is a 2 x p
            ), // index into d_phi
            rank: 0, // rank of phi (singular values > tol)
            y_est: vv,
        };
        s.set_params(&alpha);
        s
    }
}

impl LeastSquaresProblem<f64, Dynamic, Dynamic> for LevenbergMarquardtVarproProblem{
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
        // assert!(self.ind.shape() == (2, self.q));

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
        self.u = svd.u.clone().unwrap().columns(0, self.rank).into();
        self.v = svd.v_t.clone().unwrap().transpose().columns(0, self.rank).into();
        self.s = svd.singular_values.clone().rows(0, self.rank).into();


        let mut yuse = self.y.clone();
        if self.n < self.n1 {
            yuse  +=  - self.phi.column(self.n1 - 1);// % extra function Phi(:,n+1)
            panic!("This case is not actually implemented");
        }
        if self.rank < self.n {
            // println!("Warning, input functions evulations show degeneracy.");
            // println!("This leads to numerical instability.");
            // println!("Maybe try to adjust initial parameters to not be equal.");
        }
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
        // let u_s = DMatrix::<f64>::identity(self.m, self.m) - self.u.clone() * self.u.transpose();
        // jac_1 = u_s * jac_1;

        jac_1  -= self.u.clone() * (self.u.transpose()*jac_1.clone());

        // T2 = diag(1 ./ s(1:myrank)) * (V(:,1:myrank)' * T2(1:n,:));
        t_2 = nalgebra::DMatrix::<f64>::from_diagonal(&self.s.map(|x| x.recip())) * 
                    (self.v.transpose() * t_2.rows(0, self.n));
        // Jac2 = U(:,1:myrank) * T2;
        let jac_2 = self.u.clone() * t_2;

        // Jacobian = -(Jac1 + Jac2);
        Some(-(jac_1 + jac_2))
    }
     
}