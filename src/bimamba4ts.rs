struct BiMamba4TS {
    encoder: BiMambaEncoder,
    regressor: Linear,
}

impl BiMamba4TS {
    pub fn new() -> Self {
        BiMamba4TS {
            encoder: BiMambaEncoder::new(128, 256),
            regressor: Linear::new(256, 1),
        }
    }

    pub fn forward(&self, x: Tensor) -> Tensor {
        let x = self.encoder.forward(&x);
        self.regressor.forward(&x)
    }
}
