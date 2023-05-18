
// We implement a trait for every problem we want to solve
fn read_example(input:&str) -> Result<ndarray::Array2<f64>, ndarray_npy::ReadNpyError> {
    let reader = std::fs::File::open(input)?;
    let arr = Array2::<f64>::read_npy(reader)?;
    Ok(arr)
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
        let mut problem = LevenbergMarquardtVarproProblem {
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
}