use candle_core::{Module, Tensor, D};
use candle_nn as nn;

/// The Q network.
pub struct QNetV2 {
    backbone: nn::Sequential,
    value_head: nn::Sequential,
    advantage_head: nn::Sequential,
}

impl QNetV2 {
    pub fn new(
        obs_shape: candle_core::Shape,
        action_count: usize,
        vb: nn::VarBuilder,
    ) -> candle_core::Result<Self> {
        let (obs_channels, _, _) = obs_shape.dims3().unwrap();
        let b_vb = vb.pp("backbone");
        let backbone = nn::seq()
            .add(nn::conv2d(
                obs_channels,
                16,
                5,
                nn::Conv2dConfig { padding: 2, ..Default::default() },
                b_vb.pp("0"),
            )?)
            .add(nn::Activation::Silu)
            .add(conv_block_pool(16, 32, b_vb.pp("2"))?)
            .add(conv_block_pool(32, 64, b_vb.pp("5"))?)
            .add(conv_block_pool(64, 128, b_vb.pp("8"))?)
            .add(nn::conv2d(
                128,
                128,
                3,
                nn::Conv2dConfig { padding: 1, groups: 128 / 16, ..Default::default() },
                b_vb.pp("11"),
            )?)
            .add_fn(|xs| xs.max(D::Minus1)?.max(D::Minus1))
            .add(nn::Activation::Silu)
            .add(nn::linear(128, 256, b_vb.pp("15"))?)
            .add(nn::Activation::Silu);
        let value_head = nn::seq().add(nn::linear(256, 1, vb.pp("value_head").pp("0"))?);
        let advantage_head =
            nn::seq().add(nn::linear(256, action_count, vb.pp("advantage_head").pp("0"))?);

        Ok(Self { backbone, value_head, advantage_head })
    }
}

impl Module for QNetV2 {
    fn forward(&self, input_batch: &Tensor) -> candle_core::Result<Tensor> {
        let backbone_features = self.backbone.forward(input_batch)?;
        let values = self.value_head.forward(&backbone_features)?;
        let advantages = self.advantage_head.forward(&backbone_features)?;
        values.broadcast_sub(&advantages.mean(D::Minus1)?)?.broadcast_add(&advantages)
    }
}

/// Returns a convolutional block.
fn conv_block_pool(
    in_channels: usize,
    out_channels: usize,
    vb: nn::VarBuilder,
) -> candle_core::Result<nn::Sequential> {
    Ok(nn::seq()
        .add(nn::conv2d(
            in_channels,
            out_channels,
            3,
            nn::Conv2dConfig { padding: 1, ..Default::default() },
            vb,
        )?)
        .add(nn::func(|x| {
            let (_, _, w, h) = x.shape().dims4()?;
            let pad_w = w % 2;
            let pad_h = h % 2;
            x.pad_with_same(2, 0, pad_w)?.pad_with_same(3, 0, pad_h)?.max_pool2d(2)
        }))
        .add(nn::Activation::Silu))
}
