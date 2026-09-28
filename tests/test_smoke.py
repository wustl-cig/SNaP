"""CPU smoke tests: every operator trains, validates and samples end to end.

Shape/plumbing tests on random data, not quality tests. They take about a minute on a
laptop CPU and are what CI runs on every push.

    pytest -q tests/            # with pytest installed
    python tests/test_smoke.py  # same checks, no pytest needed
"""
from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from snap_unet import SNaPUNet                      # noqa: E402
from methods.registry import get_problem, _PROBLEM_REGISTRY  # noqa: E402
from pipeline import SNaPPipeline, cond_extra_channels, verify_ckpt_metadata  # noqa: E402

H = 32
DEV = torch.device("cpu")

IMAGE_PROBLEMS = {
    "inpainting": dict(mask_type="random", drop_prob=0.7, mask_channels="shared",
                       same_mask_across_batch=True, mask_seed=42, sigma=0.01),
    "box_inpainting": dict(mask_type="box", block_center=True, block_size=10,
                           mask_channels="shared", same_mask_across_batch=False, sigma=0.05),
    "denoising": dict(sigma=0.2),
    "blur_gauss": dict(sigma_blur=1.0, kernel_size=9, dim_image=H, num_channels=3, sigma=0.05),
    "sr": dict(sf=2, dim_image=H, sigma=0.05),
}


@contextmanager
def raises(exc, match=""):
    try:
        yield
    except exc as e:
        assert match in str(e), f"expected {match!r} in the error message, got: {e}"
    else:
        raise AssertionError(f"expected {exc.__name__} to be raised")


def build(stage, kwargs, source="posterior", cond_anchor="m_y", beta=0.0, C=3):
    problem = get_problem(stage)(**kwargs)
    net = SNaPUNet(input_channels=2 * C + cond_extra_channels("kspace_mask"),
                       output_channels=C, input_height=H, ch=32, ch_mult=[1, 2],
                       num_res_blocks=1, attn_resolutions=[16])
    pipe = SNaPPipeline(net, problem, tau=0.2, source_type=source, cond_anchor=cond_anchor,
                            cond_coverage="auto", p_ratio=0.5, corner_frac=0.1, beta=beta)
    pipe.solver_block = net
    return problem, net, pipe


def test_train_val_sample_every_operator():
    for stage, kwargs in IMAGE_PROBLEMS.items():
        torch.manual_seed(0)
        _, net, pipe = build(stage, kwargs)
        x0 = torch.randn(2, 3, H, H) * 0.3

        out = pipe.train_step(x0, DEV)
        assert torch.isfinite(out["loss"]), f"{stage}: non-finite loss"
        out["loss"].backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads, f"{stage}: no gradients reached the network"
        assert all(torch.isfinite(g).all() for g in grads), f"{stage}: non-finite gradients"

        val = pipe.val_step(x0, DEV)
        assert torch.isfinite(val["velocity_mse"]) and val["x_hat"].shape == x0.shape

        for k in (1, 2):
            s = pipe.sample(x0, steps=k)
            assert s["x_hat"].shape == x0.shape, f"{stage}: bad sample shape at k={k}"
            assert torch.isfinite(s["x_hat"]).all(), f"{stage}: non-finite sample at k={k}"
            assert s["y"].shape[-2:] == x0.shape[-2:], f"{stage}: display y not image-shaped"


def test_source_and_anchor_arms():
    for source in ("posterior", "gaussian"):
        for anchor in ("m_y", "aty"):
            torch.manual_seed(0)
            _, _, pipe = build("sr", IMAGE_PROBLEMS["sr"], source=source, cond_anchor=anchor)
            out = pipe.train_step(torch.randn(2, 3, H, H) * 0.3, DEV)
            assert torch.isfinite(out["loss"]), f"{source}/{anchor}: non-finite loss"
            md = pipe.metadata()
            assert md["source_type"] == source and md["cond_anchor"] == anchor


def test_whitened_metric_reduces_to_mse_at_beta0():
    torch.manual_seed(0)
    problem, _, pipe = build("denoising", IMAGE_PROBLEMS["denoising"], beta=0.0)
    obs = problem.make_observation(torch.randn(2, 3, H, H) * 0.3)
    d = torch.randn(2, 3, H, H)
    got = pipe._whitened_sq(d, obs, torch.full((2,), 0.2))
    assert torch.allclose(got, d.pow(2).flatten(1).mean(dim=1), atol=1e-6)


def test_mri_path_with_synthetic_maps():
    torch.manual_seed(0)
    problem = get_problem("cs_mri")(image_size=H, total_lines=H, acceleration_ratio=4,
                                    input_snr_db=20, cg_iters=4, mask_seed=0, random_mask=False)
    net = SNaPUNet(input_channels=4, output_channels=2, input_height=H, ch=32,
                       ch_mult=[1, 2], num_res_blocks=1, attn_resolutions=[16])
    pipe = SNaPPipeline(net, problem, tau=0.05, cond_anchor="aty", cond_coverage="auto")
    pipe.solver_block = net
    assert pipe.cond_coverage == "none", "MRI must drop the image-domain coverage channel"

    maps = torch.randn(2, 3, H, H, dtype=torch.complex64)
    maps = maps / maps.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-6)
    problem.set_maps(maps)
    x0 = torch.randn(2, 2, H, H) * 0.2
    out = pipe.train_step(x0, DEV)
    assert torch.isfinite(out["loss"])
    assert pipe.metadata()["noise_mode"] == "input_snr_db"
    assert pipe.sample(x0, steps=1)["x_hat"].shape == x0.shape


def test_checkpoint_guard_rejects_mismatched_conditioning():
    _, _, pipe = build("sr", IMAGE_PROBLEMS["sr"], cond_anchor="aty")
    for field, bad in (("cond_anchor", "m_y"), ("source_type", "gaussian")):
        saved = dict(pipe.metadata())
        saved[field] = bad
        with raises(RuntimeError, field):
            verify_ckpt_metadata(saved, pipe.metadata(), source="test")


def test_adjoint_is_the_transpose_of_forward():
    """<A x, y> == <x, A^T y> for every operator: the source's Gram solve assumes it."""
    torch.manual_seed(0)
    for stage, kwargs in IMAGE_PROBLEMS.items():
        problem = get_problem(stage)(**kwargs)
        x = torch.randn(1, 3, H, H)
        obs = problem.make_observation(torch.randn(1, 3, H, H))
        Ax = problem.forward(x, obs)
        y = torch.randn_like(Ax)
        lhs, rhs = float((Ax * y).sum()), float((x * problem.adjoint(y, obs)).sum())
        assert abs(lhs - rhs) <= 1e-3 * max(abs(lhs), abs(rhs), 1.0), \
            f"{stage}: adjoint test failed ({lhs:.6f} vs {rhs:.6f})"


def test_configs_load_and_name_a_real_problem():
    import glob
    import yaml
    files = sorted(glob.glob(os.path.join(HERE, "configs", "**", "*.yaml"), recursive=True))
    assert files, "no configs found"
    for f in files:
        cfg = yaml.safe_load(open(f))
        stage = cfg["experiment"]["stage"]
        assert stage in _PROBLEM_REGISTRY, f"{f}: unknown stage {stage!r}"
        assert stage in cfg["methods"], f"{f}: no methods.{stage} block"
        assert cfg["model"]["ch"] % 32 == 0, f"{f}: model.ch must be a multiple of 32 (GroupNorm)"


def test_manifest_matches_the_configs_it_points_at():
    import json
    import yaml
    man = json.load(open(os.path.join(HERE, "checkpoints", "manifest.json")))
    for name, e in man["models"].items():
        cfg_path = os.path.join(HERE, e["config"])
        assert os.path.exists(cfg_path), f"{name}: missing {e['config']}"
        cfg = yaml.safe_load(open(cfg_path))
        assert cfg["experiment"]["stage"] == e["stage"], f"{name}: stage disagrees with the manifest"
        assert len(e["sha256"]) == 64, f"{name}: malformed sha256"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
