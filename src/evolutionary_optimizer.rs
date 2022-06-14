use nalgebra::{DVector};

use levenberg_marquardt::{LevenbergMarquardt};

use crate::varpro::LevenbergMarquardtVarproProblem;
use oxigen::prelude::*;

// extern crate oxigen;
extern crate rand;
extern crate rand_distr;

// use rand_distr::{Normal, Distribution};
// use rand::prelude::*;
use std::fmt::Display;
use std::fs::File;
use rand::{Rng};
use rand::random;
// use oxigen::Genotype::GenotypeHash;
// use rand::prelude::SmallRng;
// use rand::FromEntropy;
// use rand::rngs::SmallRng;


#[derive(Clone, PartialEq, Debug)]
pub struct NonlinearParameters {
    pub genes: Vec<f64>,
    pub x: DVector<f64>,
    pub y: DVector<f64>,
    pub result: Option<LevenbergMarquardtVarproProblem>,
}
impl Display for NonlinearParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "{:4?}", self.genes)
    }
}

#[derive(Clone)]
pub struct Points {
    pub x: DVector<f64>,
    pub y: DVector<f64>,
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
            genes: vec![], 
            x: points.x.clone(), 
            y: points.y.clone(), 
            result: None
        };
        for _i in 0..*size {
            individual.genes.push((random::<f64>())*3.);
        }
        individual
    }

    // This function returns the maximum score possible (n, since in the
    // worst case n queens collide) minus the number of queens that collide with others
    fn fitness(&mut self) -> f64 {
        // println!("fitness {:?}", self.genes.as_slice());
        let problem = LevenbergMarquardtVarproProblem::new(
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
        if report.termination.was_successful() {
            self.result = Some(result);
            return 1_f64/report.objective_function.abs()
        }
        else  {
            return 1e-9_f64;
        }
    }

    fn mutate<R: Rng+?Sized>(&mut self, _rgen: &mut R, index: usize) {
        self.genes[index] += (random::<f64>() - 0.5)*0.50
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness > 80000_f64
    }
}

pub fn find_solution  (
    x: &Vec<f64>,
    y: &Vec<f64>,
    _std_dev: Option<&Vec<f64>>,
    _model: Option<crate::varpro::VarproModel>,
    _seed: Option<&Vec<f64>>, // suggested initial values
    _progress_log: Option<File>,
) -> (LevenbergMarquardtVarproProblem, GeneticExecution<f64, NonlinearParameters>) {
    let ps = Points {
        x: DVector::from(x.clone()), 
        y: DVector::from(y.clone())
    };
    let n_parameters: i64 = 2;
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
    let (solutions, _generation, _progress) = exec
        .run();
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
    (result, exec)
}