use crate::sra_decider::SRA_Decider;
use crate::patch_tokenizer::PatchTokenizer;
use crate::bi_mamba_encoder::BiMambaEncoder;
use crate::regressor::Regressor;
use std::rc::Rc;
use candle_core::tensor::Device;

struct BiMamba4TS {
    decider: SRA_Decider,
    tokenizer: PatchTokenizer,
    encoder: BiMambaEncoder,
    regressor: Regressor,
    device: Rc<Device>,
}

impl BiMamba4TS {
    pub fn new(device: Rc<Device>) -> Self {
        BiMamba4TS {
            decider: SRA_Decider::new(10),
            tokenizer: PatchTokenizer::new(24),
            encoder: BiMambaEncoder::new(128, 256, Rc::clone(&device)),
            regressor: Regressor::new(256, 1, Rc::clone(&device)),
            device,
        }
    }

    pub fn forward(&self, x: Tensor, correlations: &Tensor) -> Tensor {
        let strategy = self.decider.forward(correlations); // Use the strategy to adjust processing if necessary
        let x = self.tokenizer.forward(&x);
        let x = self.encoder.forward(&x);
        self.regressor.forward(&x)
    }
}

