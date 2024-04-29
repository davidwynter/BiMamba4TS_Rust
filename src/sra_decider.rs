extern crate ndarray;
extern crate ndarray_stats;

use ndarray::{Array2, ArrayView1};
use ndarray_stats::CorrelationExt;

struct SRA_Decider {
    threshold: f64,
}

impl SRA_Decider {
    fn new(threshold: f64) -> Self {
        SRA_Decider { threshold }
    }

    /// Calculate the maximum Pearson correlation coefficient between all pairs of series
    /// and decide the tokenization strategy based on the threshold.
    fn forward(&self, x: &Array2<f64>) -> usize {
        let num_series = x.nrows();
        let mut max_correlation = 0.0;

        // Iterate over pairs of series to compute the correlation coefficients
        for i in 0..num_series {
            for j in i + 1..num_series {
                let series_i = x.row(i);
                let series_j = x.row(j);
                if let Ok(correlation) = series_i.pearson_correlation(&series_j) {
                    max_correlation = max_correlation.max(correlation);
                }
            }
        }

        // Decide the strategy based on the maximum correlation found
        if max_correlation >= self.threshold {
            1 // Use channel-mixing strategy
        } else {
            0 // Use channel-independent strategy
        }
    }
}


