extern crate criterion;
use criterion::{black_box, criterion_group, criterion_main, Criterion,
BenchmarkId, Throughput};

// extern crate quick_fitting;
extern crate rand;
use nalgebra::{DVector};
use rand::{thread_rng};

extern crate rand_distr;
use quifi::varpro::NExponentialProblemVarpro;
use rand_distr::Distribution;

use levenberg_marquardt::{differentiate_numerically, 
                          LevenbergMarquardt, 
                          LeastSquaresProblem};
fn varpro_benchmark(c: &mut Criterion) {
    /*
    * VARPRO WITH LM
    */
    // const INC: u64 = 11634580027462260723;
    let dist = rand_distr::Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut routine = |m:usize| {
        let x = DVector::<f64>::from_iterator(m, (0..m).map(|x| (x as f64)/300.0));
        let y = x.map(|x| 0.001*dist.sample(&mut rng) + 3.0*(-x*0.5).exp() + 5.0*(-x*3.0).exp());
        
        let problem = NExponentialProblemVarpro::new(
            x.clone(), y.clone(),
            DVector::<f64>::from_element(x.len(), 1.0),
            DVector::<f64>::from_vec(vec![1.0, 2.0]),
            3, // n, number of linear parameters
            3, // n1, basis functions (including constant term)
        );
        LevenbergMarquardt::new().minimize(problem.clone());
    };

    let mut group = c.benchmark_group("Varpro");
    for size in (500..5000).step_by(500) {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| routine(size));
        });
    };
    group.finish();
}

criterion_group!(benches, varpro_benchmark);
criterion_main!(benches);