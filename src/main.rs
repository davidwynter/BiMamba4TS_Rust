extern crate ndarray;
extern crate ndarray_stats;

use ndarray::Array2;
use ndarray_stats::CorrelationExt;

mod sra_decider;
use sra_decider::SRA_Decider;

fn main() {
    // Example multivariate time series data
    let data = Array2::<f64>::from_shape_vec(
        (3, 5),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0,  // Series 1
            2.0, 3.0, 4.0, 5.0, 6.0,  // Series 2 (correlated with Series 1)
            10.0, 10.5, 10.5, 10.0, 10.5,  // Series 3 (less correlated with Series 1 and 2)
        ],
    ).unwrap();

    // Initialize the SRA_Decider with a high threshold to test sensitivity to strong correlations
    let decider = SRA_Decider::new(0.95); 

    // Decide on a strategy based on the calculated correlations
    let strategy = decider.forward(&data);

    // Output the chosen strategy to the console
    println!("Chosen Strategy: {}", if strategy == 1 { "Channel-Mixing" } else { "Channel-Independent" });
}
