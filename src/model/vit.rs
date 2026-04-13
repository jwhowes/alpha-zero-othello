use candle_core::{Device, IndexOp, Result, Tensor, Var};
use candle_nn::{
    LayerNorm, Linear, Module,
    ops::{silu, softmax},
};

use crate::board::GRID_SIZE;

struct Attention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,

    w_o: Linear,

    n_heads: usize,
    d_attn: usize,
    scale: Tensor,
}

impl Attention {
    fn new(d_model: usize, n_heads: usize, device: &Device) -> Result<Self> {
        let d_attn = d_model / n_heads;

        Ok(Self {
            w_q: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_model, d_model), device)?,
                None,
            ),
            w_k: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_model, d_model), device)?,
                None,
            ),
            w_v: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_model, d_model), device)?,
                None,
            ),

            w_o: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_model, d_model), device)?,
                None,
            ),

            scale: Tensor::full((1. / d_attn as f32).sqrt(), (1, 1, 1, 1), device)?,
            d_attn,
            n_heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, d_model) = x.dims3()?;

        let q = self
            .w_q
            .forward(x)?
            .reshape((b, l, self.n_heads, self.d_attn))?
            .transpose(1, 2)?;
        let k = self
            .w_k
            .forward(x)?
            .reshape((b, l, self.n_heads, self.d_attn))?;
        let v = self
            .w_v
            .forward(x)?
            .reshape((b, l, self.n_heads, self.d_attn))?
            .transpose(1, 2)?;

        let attn = q.matmul(&k)?.broadcast_mul(&self.scale)?;

        self.w_o.forward(
            &softmax(&attn, 3)?
                .matmul(&v)?
                .transpose(1, 2)?
                .reshape(&[b, l, d_model])?,
        )
    }
}

struct SwiGLU {
    hidden: Linear,
    gate: Linear,
    out: Linear,
}

impl SwiGLU {
    fn new(d_model: usize, d_hidden: Option<usize>, device: &Device) -> Result<Self> {
        let d_hidden = d_hidden.unwrap_or(4 * d_model);

        Ok(Self {
            hidden: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_hidden, d_model), device)?,
                None,
            ),
            gate: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_hidden, d_model), device)?,
                None,
            ),
            out: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (d_model, d_hidden), device)?,
                None,
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.out
            .forward(&(silu(&self.gate.forward(x)?)? * &self.hidden.forward(x)?)?)
    }
}

struct ViTBlock {
    attn: Attention,
    attn_norm: LayerNorm,

    ffn: SwiGLU,
    ffn_norm: LayerNorm,
}

impl ViTBlock {
    fn new(d_model: usize, n_heads: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(d_model, n_heads, device)?,
            attn_norm: LayerNorm::rms_norm(Tensor::randn(0f32, 0.1_f32, (d_model,), device)?, 1e-6),

            ffn: SwiGLU::new(d_model, None, device)?,
            ffn_norm: LayerNorm::rms_norm(Tensor::randn(0f32, 0.1_f32, (d_model,), device)?, 1e-6),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?)?)?;

        &x + self.ffn.forward(&self.ffn_norm.forward(&x)?)?
    }
}

pub struct ViT {
    patch_stem: Linear,
    pos_emb: Var,
    cls_emb: Var,

    blocks: Vec<ViTBlock>,

    patch_head: Linear,
    cls_head: Linear,

    patch_size: usize,
    num_patches: usize,
    d_model: usize,
}

#[derive(Clone, Copy)]
pub struct ViTConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub patch_size: usize,
}

impl ViT {
    fn new(
        image_size: usize,
        in_channels: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        patch_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let num_patches = image_size / patch_size;

        Ok(Self {
            patch_stem: Linear::new(
                Tensor::randn(
                    0f32,
                    0.1_f32,
                    (d_model, in_channels * patch_size * patch_size),
                    device,
                )?,
                None,
            ),
            pos_emb: Var::from_tensor(&Tensor::randn(
                0f32,
                0.1_f32,
                (1, num_patches * num_patches, d_model),
                device,
            )?)?,
            cls_emb: Var::from_tensor(&Tensor::randn(0f32, 0.1_f32, (1, 1, d_model), device)?)?,

            blocks: (0..n_layers)
                .map(|_| ViTBlock::new(d_model, n_heads, device))
                .collect::<Result<Vec<ViTBlock>>>()?,

            patch_head: Linear::new(
                Tensor::randn(0f32, 0.1_f32, (patch_size * patch_size, d_model), device)?,
                None,
            ),
            cls_head: Linear::new(Tensor::randn(0f32, 0.1_f32, (1, d_model), device)?, None),

            patch_size,
            num_patches,
            d_model,
        })
    }

    pub fn from_config(config: ViTConfig, device: &Device) -> Result<Self> {
        Self::new(
            GRID_SIZE,
            3,
            config.d_model,
            config.n_heads,
            config.n_layers,
            config.patch_size,
            device,
        )
    }

    /*
        Returns prior (B x h x w) and value (B)
    */
    pub fn forward(&self, image: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, c, h, w) = image.dims4()?;

        let x = self
            .patch_stem
            .forward(
                &image
                    .reshape((
                        b,
                        c,
                        self.num_patches,
                        self.patch_size,
                        self.num_patches,
                        self.patch_size,
                    ))?
                    .permute((0, 1, 2, 4, 3, 5))?
                    .flatten(1, 3)?,
            )?
            .broadcast_add(&self.pos_emb)?;

        let mut x = Tensor::cat(&[self.cls_emb.expand((b, 1, self.d_model))?, x], 1)?;

        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }

        // B x (n * n) x (p * p)
        let prior = self
            .patch_head
            .forward(&x.i((.., 1..))?)?
            .reshape((
                b,
                self.num_patches,
                self.num_patches,
                self.patch_size,
                self.patch_size,
            ))?
            .permute((0, 1, 3, 2, 4))?
            .reshape((b, h, w))?;

        let value = self.cls_head.forward(&x.i((.., 0))?)?.squeeze(1)?;

        Ok((prior, value))
    }
}
