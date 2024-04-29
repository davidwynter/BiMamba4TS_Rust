struct BiMambaEncoder {
    forward_block: MambaBlock,
    backward_block: MambaBlock,
}

impl BiMambaEncoder {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        BiMambaEncoder {
            forward_block: MambaBlock::new(in_features, out_features),
            backward_block: MambaBlock::new(in_features, out_features),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let forward_output = self.forward_block.forward(x);
        let backward_input = x.flip(1);  // TODO implement a flip method
        let backward_output = self.backward_block.forward(&backward_input);
        forward_output + backward_output
    }
}
