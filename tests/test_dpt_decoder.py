import torch

from tips_decoder.models.dpt_decoder import DPTDepthDecoder


def test_dpt_decoder_output_shape():
    decoder = DPTDepthDecoder(embed_dim=384, channels=64, output_scale=2)
    features = torch.randn(2, 384, 32, 32)
    output = decoder(features)
    assert output.shape == (2, 1, 256, 256)
