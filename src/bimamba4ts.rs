extern crate candle_core;
use candle_core::tensor::Tensor;
use std::rc::Rc;
use crate::sra_decider::SRA_Decider;
use crate::patch_tokenizer::{PatchTokenizer, TokenizationStrategy};
use crate::bi_mamba_encoder::BiMambaEncoder;
use crate::regressor::Regressor;

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
            decider: SRA_Decider::new(10, 0.6), // assuming a threshold is needed
            tokenizer: PatchTokenizer::new(24, TokenizationStrategy::ChannelIndependent), // default strategy
            encoder: BiMambaEncoder::new(128, 256, Rc::clone(&device)),
            regressor: Regressor::new(256, 1, Rc::clone(&device)),
            device,
        }
    }

    pub fn forward(&self, x: Tensor, correlations: &Tensor) -> Tensor {
        // Determine the strategy from SRA_Decider
        let strategy = if self.decider.forward(correlations) == 1 {
            TokenizationStrategy::ChannelMixing
        } else {
            TokenizationStrategy::ChannelIndependent
        };

        // Tokenize patches with the determined strategy
        let tokenized_patches = self.tokenizer.forward(&x, strategy);

        // Pass tokenized data through the encoder and regressor
        let encoded_output = self.encoder.forward(&tokenized_patches);
        self.regressor.forward(&encoded_output)
    }
}
