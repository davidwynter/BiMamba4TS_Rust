extern crate ndarray;
use ndarray::{Array3, ArrayView2, Array4, Axis, s};

pub struct PatchTokenizer {
    pub patch_size: usize,
}

impl PatchTokenizer {
    /// Constructs a new `PatchTokenizer`.
    pub fn new(patch_size: usize) -> Self {
        PatchTokenizer { patch_size }
    }

    /// Transforms the input data into patches.
    ///
    /// # Arguments
    /// * `x` - A 2D array (`ArrayView2`) where rows correspond to time series and columns to time points.
    ///
    /// # Returns
    /// * `Array4<f64>` - A 4D array where the added dimensions correspond to patches and their contents.
    ///
    /// # Panics
    /// * Panics if the number of columns in `x` is not divisible by `patch_size`.
    pub fn forward(&self, x: ArrayView2<f64>) -> Array4<f64> {
        let num_series = x.nrows();
        let sequence_length = x.ncols();

        if sequence_length % self.patch_size != 0 {
            panic!("The sequence length must be divisible by the patch size.");
        }

        let num_patches = sequence_length / self.patch_size;

        // Reshape the array to create patches
        let patched = Array3::from_shape_fn((num_series, num_patches, self.patch_size), |(i, j, k)| {
            x[[i, j * self.patch_size + k]]
        });

        // Optionally, add another dimension for patch sets if needed
        patched.into_shape((num_series, 1, num_patches, self.patch_size)).unwrap()
    }
}


