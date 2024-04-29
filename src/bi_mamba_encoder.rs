use candle_core::tensor::{Tensor, Device};
use std::rc::Rc;

/// Flips a tensor along the specified dimension.
fn flip(tensor: &Tensor, dim: usize, device: Rc<Device>) -> Tensor {
    let len = tensor.shape()[dim];
    let indices: Vec<i64> = (0..len as i64).rev().collect();
    let indices_tensor = Tensor::from_data(indices.as_slice(), device);  // Use the device passed as argument

    tensor.index_select(dim, &indices_tensor)
}


struct BiMambaEncoder {
    forward_block: MambaBlock,
    backward_block: MambaBlock,
    device: Rc<Device>,  // This should be Rc<Device> if shared between components
}

impl BiMambaEncoder {
    pub fn new(in_features: usize, out_features: usize, device: Rc<Device>) -> Self {
        BiMambaEncoder {
            forward_block: MambaBlock::new(in_features, out_features, Rc::clone(&device)),
            backward_block: MambaBlock::new(in_features, out_features, Rc::clone(&device)),
            device,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let forward_output = self.forward_block.forward(x);
        let backward_input = flip(x, 1, Rc::clone(&self.device));  // Correct usage of device
        let backward_output = self.backward_block.forward(&backward_input);
        forward_output + backward_output
    }
}
