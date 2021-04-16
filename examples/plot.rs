extern crate plotters;
use nalgebra::{DVector};

pub fn plot_results(x: &DVector<f64>, y: &DVector<f64>, yf: &DVector<f64>, path:&str)  -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("y=x^2", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(((x[0] as f32))..(x[x.len()-1] as f32), (y.min() as f32)..(y.max() as f32))?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (0..x.len()).map(|k| (x[k] as f32, y[k] as f32)),
            &RED,
        ))?
        .label("y = x^2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            (0..x.len()).map(|k| (x[k] as f32, yf[k] as f32)),
            &GREEN,
        ))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;
    Ok(())
}

fn main () {}