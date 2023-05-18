// #![allow(dead_code)]
// extern crate ndarray;
// extern crate ndarray_npy;
// // use numpy::{array, s};
// use ndarray_npy::{ReadNpyError, ReadNpyExt, WriteNpyError, WriteNpyExt};
// use ndarray::{array, Array2,s, Array};
// use std::fs::File;
// use std::io::BufWriter;

// pub fn read_npy(input:&str) -> Result<Array2<f64>, ReadNpyError> {
//     let reader = File::open(input)?;
//     let arr: Array2::<f64> = ndarray_npy::read_npy(input)?;
//     Ok(arr)
// }
// pub fn write_npy() -> Result<(), WriteNpyError> {
//     let arr: Array2<i32> = array![[1, 2, 3], [4, 5, 6]];
//     let writer = BufWriter::new(File::create("array.npy")?);
//     arr.write_npy(writer)?;
//     Ok(())
// }

// fn main() {
    // let arr = read_npy("data/tesdata.npy").unwrap();
    // let arr0 = arr.slice(s![0, ..]);
    // print!("arr =\n{}", arr0);
// }