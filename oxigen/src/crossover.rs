//! This module contains the Crossover trait and the provided crossover functions.

use genotype::Genotype;
use std::cmp::{min, PartialEq};
use CrossoverFunctions::*;
use rand_distr::{Distribution, Uniform};
use rand::{thread_rng, Rng};

/// This trait defines the cross function.
pub trait Crossover<T: PartialEq, G: Genotype<T>>: Send + Sync {
    /// Generates two children combining the two selected individuals.
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G);
}

/// Provided crossover functions.
#[derive(Debug)]
pub enum CrossoverFunctions {
    /// Single point Crossover.
    SingleCrossPoint,
    /// Multi point Crossover.
    MultiCrossPoint,
    /// Uniform Crossover.
    UniformCross,
}

impl<T: PartialEq, G: Genotype<T>> Crossover<T, G> for CrossoverFunctions {
    #[allow(clippy::comparison_chain)]
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G) {
        match self {
            SingleCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                if ind_size == 0 {
                    panic!("The size of the smallest individual is 0");
                } else if ind_size == 1 {
                    return crosspoint_cross_single_genes(&ind1, &ind2);
                }
                // let cross_point = SmallRng::from_entropy().sample(Uniform::from(1..ind_size));
                let cross_point = Uniform::from(0..ind_size).sample(&mut thread_rng());

                let mut child1 = ind1.clone();
                child1.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i < cross_point { gen1 } else { gen2 }),
                );
                let mut child2 = ind2.clone();
                child2.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i < cross_point { gen2 } else { gen1 }),
                );

                (child1, child2)
            }
            MultiCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                if ind_size == 0 {
                    panic!("The size of the smallest individual is 0");
                } else if ind_size == 1 {
                    return crosspoint_cross_single_genes(&ind1, &ind2);
                }
                let mut cross_points = Vec::new();
                let mut point_maximum = ind_size / 2;
                if point_maximum <= 2 {
                    point_maximum = 3.min(ind_size);
                }
                // let mut i = SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                let mut i = Uniform::from(1..point_maximum).sample(&mut thread_rng());
                while i < ind_size {
                    cross_points.push(i);
                    // i += SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                    i += Uniform::from(1..point_maximum).sample(&mut thread_rng());
                }
                cross_points.push(ind_size);

                let mut child1 = ind1.clone();
                child1.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen2;
                                    } else {
                                        return gen1;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen1
                        }),
                );
                let mut child2 = ind2.clone();
                child2.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen1;
                                    } else {
                                        return gen2;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen2
                        }),
                );

                (child1, child2)
            }
            UniformCross => {
                // Elements that change (only until the shortest individual)
                // As nth consumes the iterator, besides the global index the
                // difference with the previous (+ 1 for the taken value) change is used
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                let mut change: Vec<(usize, usize)> = Vec::with_capacity(ind_size);
                let mut rng = rand::thread_rng();
                let mut previous = 0;
                for i in 0..ind_size {
                    if rng.gen() {
                        change.push((i, i - previous));
                        previous = i + 1;
                    }
                }
                if !change.is_empty() {
                    let mut other = ind2.clone().into_iter();
                    // change must be cloned to use it with the second child without removed items
                    let mut change1 = change.clone();
                    let mut child1 = ind1.clone();
                    child1.from_iter(child1.clone().into_iter().enumerate().map(|(i, gen)| {
                        if !change1.is_empty() && change1[0].0 == i {
                            other.nth(change1.remove(0).1).unwrap()
                        } else {
                            gen
                        }
                    }));
                    let mut other = ind1.clone().into_iter();
                    let mut child2 = ind2.clone();
                    child2.from_iter(ind2.clone().into_iter().enumerate().map(|(i, gen)| {
                        if !change.is_empty() && change[0].0 == i {
                            other.nth(change.remove(0).1).unwrap()
                        } else {
                            gen
                        }
                    }));

                    (child1, child2)
                } else {
                    // No changes
                    (ind1.clone(), ind2.clone())
                }
            }
        }
    }
}

/// Crosspoint crossover when one or both individuals have length 1
fn crosspoint_cross_single_genes<T: PartialEq, G: Genotype<T>>(ind1: &G, ind2: &G) -> (G, G) {
    let len1 = ind1.iter().len();
    let len2 = ind2.iter().len();

    if len1 > 1 {
        // interchange ind2 gene with a random gene in ind1
        interchange_gene(&ind2, &ind1, len1)
    } else if len2 > 1 {
        // interchange ind2 gene with a random gene in ind1
        interchange_gene(&ind1, &ind2, len2)
    } else {
        // children equal to parents, since both have length 1
        (ind2.clone(), ind1.clone())
    }
}

/// Interchange len1_ind gene into a random position of the another individual
fn interchange_gene<T: PartialEq, G: Genotype<T>>(
    len1_ind: &G,
    bigger_ind: &G,
    bigger_len: usize,
) -> (G, G) {
    // let interchanged = SmallRng::from_entropy().sample(Uniform::from(0..bigger_len));
    let interchanged = Uniform::from(0..bigger_len).sample(&mut thread_rng());
    // return the interchanged gene of bigger_ind as child1 and the bigger_ind
    // with the len1_ind gene in the interchanged position as child2
    let mut child1 = len1_ind.clone();
    let mut child2 = bigger_ind.clone();
    child1.from_iter(
        bigger_ind
            .clone()
            .into_iter()
            .enumerate()
            .filter(|(i, _gen)| *i == interchanged)
            .map(|(_i, gen)| gen),
    );
    child2.from_iter(bigger_ind.clone().into_iter().enumerate().map(|(i, gen)| {
        if i == interchanged {
            len1_ind.clone().into_iter().next().unwrap()
        } else {
            gen
        }
    }));
    (child1, child2)
}
