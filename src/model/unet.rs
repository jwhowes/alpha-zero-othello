use candle_core::{Device, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, GroupNorm, Linear, Module,
    ops::silu,
};
use serde::{Deserialize, Serialize};
use std::iter::zip;

struct ResNetBlock {
    hidden: Conv2d,
    norm: GroupNorm,
    out: Conv2d,
}

impl ResNetBlock {
    fn new(d_model: usize, device: &Device) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        Ok(Self {
            hidden: Conv2d::new(
                Tensor::randn(0f32, 0.1, (d_model, d_model, 3, 3), device)?,
                None,
                conv_config,
            ),
            norm: GroupNorm::new(
                Tensor::randn(0f32, 0.1, (d_model,), device)?,
                Tensor::randn(0f32, 0.1, (d_model,), device)?,
                d_model,
                32,
                1e-6,
            )?,
            out: Conv2d::new(
                Tensor::randn(0f32, 0.1, (d_model, d_model, 3, 3), device)?,
                None,
                conv_config,
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x + self
            .out
            .forward(&silu(&self.norm.forward(&self.hidden.forward(x)?)?)?)?
    }
}

pub struct UNet {
    stem: Conv2d,

    down_path: Vec<Vec<ResNetBlock>>,
    down_samples: Vec<Conv2d>,

    mid_blocks: Vec<ResNetBlock>,

    up_samples: Vec<ConvTranspose2d>,
    up_combines: Vec<Conv2d>,
    up_path: Vec<Vec<ResNetBlock>>,

    value_head: Linear,
    prior_head: Conv2d,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct UNetConfig {
    dims: Vec<usize>,
    depths: Vec<usize>,
}

impl UNet {
    pub fn from_config(UNetConfig { dims, depths }: UNetConfig, device: &Device) -> Result<Self> {
        let n = dims.len();

        let mut down_path = Vec::with_capacity(n - 1);
        let mut down_samples = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            down_path.push(
                (0..depths[i])
                    .map(|_| ResNetBlock::new(dims[i], device))
                    .collect::<Result<Vec<_>>>()?,
            );

            down_samples.push(Conv2d::new(
                Tensor::randn(0f32, 0.1, (dims[i], dims[i + 1], 2, 2), device)?,
                None,
                Conv2dConfig {
                    stride: 2,
                    ..Default::default()
                },
            ));
        }

        let mut up_samples = Vec::with_capacity(n - 1);
        let mut up_combines = Vec::with_capacity(n - 1);
        let mut up_path = Vec::with_capacity(n - 1);

        for i in (1..n - 1).rev() {
            up_samples.push(ConvTranspose2d::new(
                Tensor::randn(0f32, 0.1, (dims[i - 1], dims[i], 2, 2), device)?,
                None,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
            ));

            up_combines.push(Conv2d::new(
                Tensor::randn(0f32, 0.1, (2 * dims[i], dims[i], 1, 1), device)?,
                None,
                Conv2dConfig::default(),
            ));

            up_path.push(
                (0..depths[i])
                    .map(|_| ResNetBlock::new(dims[i], device))
                    .collect::<Result<Vec<_>>>()?,
            );
        }

        Ok(Self {
            stem: Conv2d::new(
                Tensor::randn(0f32, 0.1, (dims[0], 3, 1, 1), device)?,
                None,
                Conv2dConfig::default(),
            ),

            down_path,
            down_samples,

            mid_blocks: (0..depths[depths.len() - 1])
                .map(|_| ResNetBlock::new(dims[dims.len() - 1], device))
                .collect::<Result<Vec<_>>>()?,

            value_head: Linear::new(
                Tensor::randn(0f32, 0.1, (1, dims[dims.len() - 1]), device)?,
                None,
            ),

            up_samples,
            up_combines,
            up_path,

            prior_head: Conv2d::new(
                Tensor::randn(0f32, 0.1, (2, dims[0], 1, 1), device)?,
                None,
                Conv2dConfig::default(),
            ),
        })
    }

    pub fn new(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = self.stem.forward(x)?;

        let mut activations = Vec::with_capacity(self.down_path.len());

        for (blocks, sample) in zip(self.down_path.iter(), self.down_samples.iter()) {
            for block in blocks.iter() {
                x = block.forward(&x)?;
            }

            activations.push(x.clone());

            x = sample.forward(&x)?;
        }

        for block in self.mid_blocks.iter() {
            x = block.forward(&x)?;
        }

        let value = self.value_head.forward(&x.mean((2, 3))?)?;

        for ((blocks, act), (sample, combine)) in zip(
            zip(self.up_path.iter(), activations.into_iter()),
            zip(self.up_samples.iter(), self.up_combines.iter()),
        ) {
            x = combine.forward(&Tensor::cat(&[sample.forward(&x)?, act], 1)?)?;

            for block in blocks {
                x = block.forward(&x)?;
            }
        }

        Ok((self.prior_head.forward(&x)?, value))
    }
}
