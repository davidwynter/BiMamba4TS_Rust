extern crate ndarray;

use ndarray::Array2;
use super::*; // import everything from the parent module

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_correlation() {
        let data = Array2::<f64>::from_shape_vec(
            (3, 5),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0,
                2.0, 3.0, 4.0, 5.0, 4.0,
                1.0, 1.1, 1.2, 1.1, 1.0,
            ],
        ).unwrap();

        let decider = SRA_Decider::new(0.9);
        let strategy = decider.forward(&data);

        // Here we expect "Channel-Mixing" due to a high threshold
        assert_eq!(strategy, 1, "Expected channel-mixing strategy for high correlation threshold.");
    }

    #[test]
    fn test_low_correlation() {
        let data = Array2::<f64>::from_shape_vec(
            (3, 5),
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0,
                2.0, 2.0, 2.0, 2.0, 2.0,
                3.0, 3.0, 3.0, 3.0, 3.0,
            ],
        ).unwrap();

        let decider = SRA_Decider::new(0.1); // Set a very low threshold for correlation
        let strategy = decider.forward(&data);

        // Here we expect "Channel-Independent" since data is not correlated
        assert_eq!(strategy, 0, "Expected channel-independent strategy for low correlation threshold.");
    }
}
