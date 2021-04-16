#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

extern crate ndarray;
extern crate ndarray_npy;
extern crate plotters;
use ndarray::{Array2,s};
use ndarray_npy::ReadNpyExt;
use nalgebra::{DVector,DMatrix};

mod optimizer;
use optimizer::fit_biexponential;

mod plot;
use plot::plot_results;

extern crate oxigen;
extern crate rand;

use oxigen::prelude::*;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::fmt::Display;
use std::fs::File;
// use oxigen::Genotype::GenotypeHash;
// use rand::prelude::SmallRng;
use std::slice::Iter;
use std::vec::IntoIter;
// use rand::FromEntropy;
// use rand::rngs::SmallRng;

#[derive(Clone, PartialEq, Eq, std::hash::Hash)]
struct QueensBoard {
    genes: Vec<u8>,
}
impl Display for QueensBoard {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        let mut s = String::new();
        for row in self.iter() {
            let mut rs = String::from("|");
            for i in 0..self.genes.len() {
                if i == *row as usize {
                    rs.push_str("Q|");
                } else {
                    rs.push_str(" |")
                }
            }
            rs.push('\n');
            s.push_str(&rs);
        }
        write!(f, "{}", s)
    }
}
impl Genotype<u8> for QueensBoard {
    type ProblemSize = u8;
    type ArgumentType = ();

    fn iter(&self) -> std::slice::Iter<u8> {
        self.genes.iter()
    }
    fn into_iter(self) -> std::vec::IntoIter<u8> {
        self.genes.into_iter()
    }
    fn from_iter<I: Iterator<Item = u8>>(&mut self, genes: I) {
        self.genes = genes.collect();
    }

    fn generate(size: &Self::ProblemSize, arg: &()) -> Self {
        let mut individual = Vec::with_capacity(*size as usize);
        let mut rgen = SmallRng::from_entropy();
        for _i in 0..*size {
            individual.push(rgen.sample(Uniform::from(0..*size)));
        }
        QueensBoard {
            genes: individual
        }
    }

    // This function returns the maximum score possible (n, since in the
    // worst case n queens collide) minus the number of queens that collide with others
    fn fitness(&self) -> f64 {
        let size = self.genes.len();
        let diags_exceed = size as isize - 1_isize;
        let mut collisions = Vec::with_capacity(size);
        let mut verticals: Vec<isize> = Vec::with_capacity(size);
        let mut diagonals: Vec<isize> = Vec::with_capacity(size + diags_exceed as usize);
        let mut inv_diags: Vec<isize> = Vec::with_capacity(size + diags_exceed as usize);
        for _i in 0..size {
            verticals.push(-1);
            diagonals.push(-1);
            inv_diags.push(-1);
            collisions.push(false);
        }
        for _i in 0..diags_exceed as usize {
            diagonals.push(-1);
            inv_diags.push(-1);
        }

        for (row, queen) in self.iter().enumerate() {
            let mut collision = verticals[*queen as usize];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            verticals[*queen as usize] = row as isize;

            // A collision exists in the diagonal if col-row have the same value
            // for more than one queen
            let diag = ((*queen as isize - row as isize) + diags_exceed) as usize;
            collision = diagonals[diag];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            diagonals[diag] = row as isize;

            // A collision exists in the inverse diagonal if n-1-col-row have the
            // same value for more than one queen
            let inv_diag =
                ((diags_exceed - *queen as isize - row as isize) + diags_exceed) as usize;
            collision = inv_diags[inv_diag];
            if collision > -1 {
                collisions[row] = true;
                collisions[collision as usize] = true;
            }
            inv_diags[inv_diag] = row as isize;
        }

        (size - collisions.into_iter().filter(|r| *r).count()) as f64
    }

    fn mutate(&mut self, rgen: &mut SmallRng, index: usize) {
        self.genes[index] = rgen.sample(Uniform::from(0..self.genes.len())) as u8;
    }

    fn is_solution(&self, fitness: f64) -> bool {
        fitness as usize == self.genes.len()
    }

    // fn hash(&self) -> Self {
    //     self.clone()
    // }
}


fn read_example(input:&str) -> Result<ndarray::Array2<f64>, ndarray_npy::ReadNpyError> {
    let reader = std::fs::File::open(input)?;
    let arr = Array2::<f64>::read_npy(reader)?;
    Ok(arr)
}


fn main() {
    let n_queens: u8 = 10;

    let progress_log = File::create("data/progress.csv").expect("Error creating progress log file");
    let population_log = File::create("data/population.txt").expect("Error creating population log file");
    let log2 = (f64::from(n_queens) * 4_f64).log2().ceil();

    let population_size = 2_i32.pow(log2 as u32) as usize;

    let (solutions, generation, progress, population) = GeneticExecution::<u8, QueensBoard>::new()
        .population_size(population_size)
        .genotype_size(n_queens as u8)
        .mutation_rate(Box::new(MutationRates::Linear(SlopeParams {
            start: f64::from(n_queens) / (8_f64 + 2_f64 * log2) / 100_f64,
            bound: 0.005,
            coefficient: -0.0002,
        })))
        .selection_rate(Box::new(SelectionRates::Linear(SlopeParams {
            start: log2 - 2_f64,
            bound: log2 / 1.5,
            coefficient: -0.0005,
        })))
        .select_function(Box::new(SelectionFunctions::Cup))
        .age_function(Box::new(AgeFunctions::Quadratic(
            AgeThreshold(50),
            AgeSlope(1_f64),
        )))
        .progress_log(20, progress_log)
        .population_log(2000, population_log)
        .run();
    println!("{}", solutions[0]);
    // let arr = read_example("data/tesdata.npy").unwrap();
    // // println!("@len={}\n", );
    // let n = arr.shape()[0];
    // let mut v_error = DVector::<f64>::from_element(n, 0.0);
    // for k in 0..n {
    //     let yv = arr.slice(s![k, ..]).to_vec();
    //     let xv = ndarray::Array::linspace(0., 10., 146).to_vec();
    //     let x = nalgebra::DVector::<f64>::from_vec(xv);
    //     let y = nalgebra::DVector::<f64>::from_vec(yv);
    //     let (s, p) = fit_biexponential(x, y, &[1.0, 2.0]);
    //     v_error[k] = s;
    // }
    // plot_results(DVector::<f64>::from_iterator(n, (0..n).map(|x| x as f64)), v_error.clone(), v_error.clone(), "data/f_error.png");
}