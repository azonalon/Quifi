#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2,s};
use ndarray_npy::ReadNpyExt;
use nalgebra::{DVector,DMatrix,VecStorage, Dynamic, U1};

use levenberg_marquardt::{differentiate_numerically,differentiate_holomorphic_numerically};
use levenberg_marquardt::{LeastSquaresProblem,LevenbergMarquardt};

mod least_squares;
use least_squares::NExponentialProblemVarpro;

// extern crate oxigen;
extern crate rand;
extern crate rand_distr;

use oxigen::prelude::*;
use rand_distr::{Normal, Distribution};
// use rand::prelude::*;
use std::fmt::Display;
use std::fs::File;
use rand::prelude::SmallRng;
// use oxigen::Genotype::GenotypeHash;
// use rand::prelude::SmallRng;
use rand::random;
use std::slice::Iter;
use std::vec::IntoIter;
// use rand::FromEntropy;
// use rand::rngs::SmallRng;


#[derive(Clone, PartialEq, Debug)]
struct NonlinearParameters {
    genes: Vec<f64>,
    x: DVector<f64>,
    y: DVector<f64>,
    result: Option<NExponentialProblemVarpro>,
}
impl Display for NonlinearParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:4?}", self.genes)
    }
}

fn exp_decay(x :&DVector<f64>, tau : f64) -> DVector<f64> {
//   x.map(|x|(-x/tau).exp())
  x.map(|x| (-x*tau).exp())
}

fn exp_decay_dtau(x: &DVector<f64>,tau: f64) -> DVector<f64> {
//   tvec.map(|t| (-t / tau).exp() * t / tau.powi(2))
  x.map(|x| -x*(-x * tau).exp() )
}

#[derive(Clone)]
struct Points {
    x: DVector<f64>,
    y: DVector<f64>,
}
impl Default for Points {
    // fn default() -> Self {Points{x: vec![], y: vec![] }}
    fn default() -> Self {Points{x: DVector::<f64>::zeros(1), y: DVector::<f64>::zeros(1)}}
}

impl Genotype<f64> for NonlinearParameters {
    type ProblemSize = i64;
    type ArgumentType = Points;

    fn iter(&self) -> std::slice::Iter<f64> {
        self.genes.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<f64> {
        self.genes.into_iter()
    }
    fn from_iter<I: Iterator<Item = f64>>(&mut self, genes: I) {
        self.genes = genes.collect();
    }

    fn generate(size: &Self::ProblemSize, points: &Points) -> Self {
        // 2. Cast the fitting problem as a nonlinear least squares minimization problem
        let mut individual = NonlinearParameters {
            genes: vec![], x: points.x.clone(), 
            y: points.y.clone(), result: None
        };
        // let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.genes.push((random::<f64>()-0.5)*10.);
        }
        // println!("generated v with l = {}", individual.genes.len());
        individual
    }

    // This function returns the maximum score possible (n, since in the
    // worst case n queens collide) minus the number of queens that collide with others
    fn fitness(&mut self) -> f64 {
        // println!("fitness {:?}", self.genes.as_slice());
        let problem = NExponentialProblemVarpro::new(
            self.x.clone(),
            self.y.clone(),
            DVector::<f64>::from_element(self.x.len(), 1.0),
            DVector::<f64>::from(self.genes.clone()),
            self.genes.len() + 1, // n, number of linear parameters
            self.genes.len() + 1, // n1, basis functions (including constant term)
        );
        let (result, report) = 
            LevenbergMarquardt::new()
                                .with_patience(50)
                                // .with_stepbound(5_f64)
                                .minimize(problem);
        // println!("fitness returned with {}", report.number_of_evaluations);
        // println!("fitness {}", report.objective_function);
        if report.termination.was_successful() {
            // println!("VarPro did     terminate successfully");
            self.result = Some(result);
            return 1_f64/report.objective_function.abs()
        }
        else  {
            // println!("VarPro did not terminate successfully {:?}", 
                //  report.termination);
            return 1e-9_f64;
        }
    }

    fn mutate(&mut self, rgen: &mut SmallRng, index: usize) {
        self.genes[index] += (random::<f64>() - 0.5)*8.5
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness > 70000_f64
    }
}


fn read_example(input:&str) -> Result<ndarray::Array2<f64>, ndarray_npy::ReadNpyError> {
    let reader = std::fs::File::open(input)?;
    let arr = Array2::<f64>::read_npy(reader)?;
    Ok(arr)
}


fn main() {

    let arr = read_example("data/tesdata.npy").unwrap();
    // let mut v_error = DVector::<f64>::from_element(n, 0.0);
    // for k in 0..n {
    for q in 0..arr.shape()[0] {
        let yv = arr.slice(s![q, ..]).to_vec();
        let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
        let x = DVector::<f64>::from(xv);
        let y = DVector::<f64>::from(yv);
        let ps = Points {x, y};
        let progress_log = File::create("data/progress2.csv").expect("Error creating progress log file");
        let n_parameters: i64 = 2;
        let log2 = ((n_parameters as f64) * 4_f64).log2().ceil();
        let population_log = File::create("data/population.txt").expect("Error creating population log file");
        let population_size = 2_i32.pow(log2 as u32) as usize;
        let nt = NTournaments(population_size/2);
        let (solutions, generation, progress, population) = GeneticExecution::<f64, NonlinearParameters>::new()
            .population_size(population_size)
            .genotype_size(n_parameters)
            .fitness_arguments(ps)
            .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
                start: (n_parameters as f64) / (8_f64 + 2_f64 * log2) / 100_f64,
                bound: 0.10,
                coefficient: -0.0002,
            })))
            // .stop_criterion(Box::new(StopCriteria::SolutionsFoundOrGeneration(1, 20)))
            .stop_criterion(Box::new(StopCriteria::Generation(20)))
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
            // .progress_log(20, progress_log)
            // .population_log(2000, population_log)
            .run();
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

        println!("1 = {}", q);
        println!("solution alpha: {:3}", sol.result.clone().unwrap().alpha);
        println!("solution     c: {:3}", sol.result.clone().unwrap().c);
        println!("genes: {:.3?}", sol.genes);
        println!("solution sigma {}", sol.result.clone().unwrap().wresid.unwrap().norm_squared());
    }
    // for s in population.iter() {
    //     println!("pop: {}", s.ind.fitness());
    //     println!("pop: {:?}", s.ind.genes);
    // }
    // for s in solutions.iter() {
    //     println!("solution alpha: {:3}", s.result.clone().unwrap().alpha);
    //     println!("solution     c: {:3}", s.result.clone().unwrap().c);
    //     println!("genes: {:.3?}", s.genes);
    // }

    // println!("{}", solutions[0]);
    // let arr = read_example("data/tesdata.npy").unwrap();
    // println!("@len={}\n", );
    // let n = arr.shape()[0];
    // let mut v_error = DVector::<f64>::from_element(n, 0.0);
    // for k in 0..n {
        // let yv = arr.slice(s![k, ..]).to_vec();
        // let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
        // X = xv.clone();
        // Y = yv.clone();
        // let (s, p) = fit_biexponential(X.clone(), Y.clone(), &[1.0, 2.0]);
        // v_error[k] = s;
    // }
    // plot_results(DVector::<f64>::from_iterator(n, (0..n).map(|x| x as f64)), v_error.clone(), v_error.clone(), "data/f_error.png");
}