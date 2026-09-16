import torch
import torch.nn as nn

from ..src.vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC


def JEPApredictortest():
    predictor = VisionTransformerPredictorAC(img_size = 256, num_frames=14, patch_size = 16, embed_dim = 1408, action_embed_dim = 14)
    tokens = torch.randn(4, 7*256, 1408)
    actions = torch.randn(4, 7, 14)
    states = torch.randn(4, 7, 14)

    predicted = predictor(tokens, actions, states)

    assert predicted.shape == (4, 7*256, 1408)

    predicted.square().mean().backward()

    predicted_a = predictor(tokens, actions, states)
    predicted_b = predictor(tokens, actions.flip(1), states)

    assert not torch.allclose(predicted_a, predicted_b)


if __name__ == "__main__":
    JEPApredictortest()
    print("test passed")