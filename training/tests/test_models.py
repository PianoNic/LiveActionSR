import pytest
import torch

from lasr.models import PRESETS, LumaSR, bilinear_up2, build_model, count_params, plain_param_count


@pytest.mark.parametrize(("preset", "expected"), [("balanced", 12_340), ("quality", 47_716)])
def test_preset_param_counts(preset, expected):
    model = build_model({"preset": preset})
    assert count_params(model) == expected == plain_param_count(**PRESETS[preset])


def test_param_counts_in_target_ranges():
    assert 10_000 <= count_params(build_model({"preset": "balanced"})) <= 16_000
    assert 40_000 <= count_params(build_model({"preset": "quality"})) <= 60_000


def test_output_shape():
    model = build_model({"preset": "balanced"})
    out = model(torch.rand(2, 1, 12, 20))
    assert out.shape == (2, 1, 24, 40)


def test_residual_path_is_bilinear_when_body_is_zero():
    model = LumaSR(8, 2)
    with torch.no_grad():
        for p in model.convs[-1].parameters():
            p.zero_()
    x = torch.rand(1, 1, 10, 14)
    assert torch.allclose(model(x), bilinear_up2(x))


def test_fresh_model_starts_close_to_bilinear():
    torch.manual_seed(0)
    model = build_model({"preset": "balanced"})
    x = torch.rand(1, 1, 16, 16)
    assert (model(x) - bilinear_up2(x)).abs().mean() < 0.1


def test_bilinear_matches_texture_sampling_weights():
    x = torch.tensor([[[[0.0, 1.0]]]])  # 1x2 image
    up = bilinear_up2(x)[0, 0, 0]
    # sample positions -0.25, 0.25, 0.75, 1.25 with clamp-to-edge
    assert torch.allclose(up, torch.tensor([0.0, 0.25, 0.75, 1.0]))


def test_repconv_fuses_exactly():
    torch.manual_seed(0)
    model = LumaSR(8, 2, rep=True)
    for p in model.parameters():  # make every branch matter
        p.data.add_(0.05 * torch.randn_like(p))
    plain = model.reparameterize()
    assert not plain.rep
    assert count_params(plain) == plain_param_count(8, 2)
    x = torch.rand(2, 1, 9, 11)
    assert torch.allclose(model(x), plain(x), atol=1e-5)
