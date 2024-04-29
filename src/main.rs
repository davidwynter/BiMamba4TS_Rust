extern crate ndarray;

use crate::bi_mamba4ts::BiMamba4TS;
use ndarray::Array2;

mod bi_mamba4ts;
mod bi_mamba_encoder;
mod mamba_block;
mod patch_tokenizer;
mod sra_decider;

fn main() {
    // Example of initializing the BiMamba4TS model
    let model = BiMamba4TS::new();

    // Simulating loading or generating some input data and correlations
    // For this example, let's use random data (note: in practice, you'd likely load this from a dataset)
    let input_data = Array2::<f64>::random((10, 128), ndarray::rand::distributions::Uniform::new(0.0, 1.0));
    let correlations = Array2::<f64>::random((10, 10), ndarray::rand::distributions::Uniform::new(-1.0, 1.0));

    // Running the forward pass of the model
    let output = model.forward(input_data.view(), correlations.view());

    // Printing the output
    println!("Output from BiMamba4TS Model: {:?}", output);
}

