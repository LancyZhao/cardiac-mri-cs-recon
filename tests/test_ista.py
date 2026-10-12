"""Numerical regressions for batching, sampling gradients and resume."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ista_net import (
    ISTANetPlus,
    LearnableMask,
    cartesian_dc_gradient,
    cartesian_hard_dc,
    dc_gradient,
    hard_dc,
    simulate_cartesian_measurement,
    simulate_measurement,
)
from sampling import cartesian_mask
from train_ista import ReconSet, evaluate, run_ratio, atomic_checkpoint


class Numerics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.gt = torch.rand(4, 1, 32, 32)
        self.mask = torch.from_numpy(cartesian_mask(32, 0.3))

    def test_batch_permutation_and_numpy_measurement(self):
        zf, k = simulate_measurement(self.gt, self.mask)
        for i in range(4):
            ref = np.fft.fftshift(np.fft.fft2(self.gt[i, 0].numpy())) * self.mask.numpy()
            np.testing.assert_allclose(k[i, 0].numpy(), ref.real, atol=1e-4)
            np.testing.assert_allclose(k[i, 1].numpy(), ref.imag, atol=1e-4)
            torch.testing.assert_close(
                dc_gradient(zf, k, self.mask)[i:i+1],
                dc_gradient(zf[i:i+1], k[i:i+1], self.mask),
            )
            torch.testing.assert_close(
                hard_dc(zf, k, self.mask)[i:i+1],
                hard_dc(zf[i:i+1], k[i:i+1], self.mask),
            )

    def test_dc_matches_autograd_and_descends(self):
        _, k = simulate_measurement(self.gt, self.mask)
        x = torch.rand_like(self.gt, requires_grad=True)
        fx = torch.fft.fftshift(torch.fft.fft2(x[:, 0]), dim=(-2, -1))
        residual = fx * self.mask - torch.complex(k[:, 0], k[:, 1])
        loss = residual.abs().square().sum() / (2 * 32 * 32)
        grad, = torch.autograd.grad(loss, x)
        torch.testing.assert_close(grad, dc_gradient(x, k, self.mask))
        update = x - grad
        self.assertLess(float((update-self.gt).square().mean().detach()),
                        float((x-self.gt).square().mean().detach()))

    def test_fast_cartesian_dc_is_exact(self):
        zf_2d, k_2d = simulate_measurement(self.gt, self.mask)
        zf_1d, k_1d = simulate_cartesian_measurement(self.gt, self.mask)
        torch.testing.assert_close(zf_1d, zf_2d, rtol=1e-5, atol=1e-5)
        x = torch.rand_like(self.gt)
        torch.testing.assert_close(
            cartesian_dc_gradient(x, k_1d, self.mask),
            dc_gradient(x, k_2d, self.mask),
            rtol=1e-5,
            atol=1e-5,
        )
        net = ISTANetPlus(3, 4)
        p2, s2 = net(zf_2d, self.mask, k_2d)
        p1, s1 = net(zf_1d, self.mask, k_1d, fast_cartesian=True)
        torch.testing.assert_close(p1, p2, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(s1, s2, rtol=1e-4, atol=1e-6)
        grads = []
        for fast in (False, True):
            lm = LearnableMask(32, 0.3, self.mask)
            mask = lm()
            measure = simulate_cartesian_measurement if fast else simulate_measurement
            z, k = measure(self.gt, mask)
            p, s = net(z, mask, k, fast_cartesian=fast)
            loss = (p - self.gt).square().mean() + 0.01*s
            grads.append(torch.autograd.grad(loss, lm.logits)[0])
        torch.testing.assert_close(grads[0], grads[1], rtol=2e-4, atol=1e-7)
        torch.testing.assert_close(
            cartesian_hard_dc(x, k_1d, self.mask),
            hard_dc(x, k_2d, self.mask),
            rtol=1e-5,
            atol=1e-5,
        )
    def test_budget_acs_and_joint_gradient(self):
        for ratio in (0.1, 0.2, 0.3, 1.0):
            fixed = torch.from_numpy(cartesian_mask(32, ratio))
            lm = LearnableMask(32, ratio, fixed)
            mask = lm()
            torch.testing.assert_close(mask, fixed)
            self.assertTrue(torch.all((mask == 0) | (mask == 1)))
            self.assertEqual(int(mask[0].sum()), round(32 * ratio))
            self.assertLess(float(lm.rate_penalty().detach()), 1e-10)
            zf, k = simulate_measurement(self.gt, mask)
            net = ISTANetPlus(2, 4)
            pred, sym = net(zf, mask, k)
            ((pred-self.gt).abs().mean() + 0.01*sym).backward()
            self.assertTrue(torch.isfinite(lm.logits.grad).all())
            self.assertEqual(float(lm.logits.grad[lm.acs_lines].abs().sum()), 0.0)
            if ratio < 1:
                self.assertGreater(float(lm.logits.grad.abs().sum()), 0.0)
            self.assertEqual(lm.summary()["hard_lines"], round(32*ratio))

    def test_mask_support_can_change(self):
        lm = LearnableMask(32, 0.3, self.mask)
        initial = lm.hard_mask().clone()
        target = torch.zeros(32)
        target[:lm.outer_budget] = 1
        opt = torch.optim.Adam(lm.parameters(), lr=0.1)
        for _ in range(30):
            opt.zero_grad()
            loss = -(lm()[0] * target).sum() + 10 * lm.rate_penalty()
            loss.backward()
            opt.step()
        self.assertFalse(torch.equal(initial, lm.hard_mask()))
        self.assertEqual(int(lm.hard_mask()[0].sum()), 10)
        self.assertTrue(lm.hard_mask()[0, lm.acs_lines].all())


class Workflow(unittest.TestCase):
    def test_resume_and_no_dc_evaluation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(0)
            for name, n in (("train", 8), ("test", 4)):
                np.savez(root / f"{name}.npz", img=rng.random((n, 32, 32), dtype=np.float32))
            meta = {"size": 32, "split": {"train": ["train"], "test": ["test"]},
                    "n_cases": {"train": 1, "test": 1}, "n_slices": {"train": 8, "test": 4}}
            (root / "meta.json").write_text(json.dumps(meta))
            args = argparse.Namespace(
                out=str(root / "whole"), mask_mode="learnable", phases=1,
                filters=2, epochs=2, batch=4, lr=1e-3, mask_lr=1e-3,
                sym_w=0.01, mask_gamma=10.0, acs_frac=0.25, seed=0,
                mask_seed=0, final_hard_dc=False, resume=False,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                complete = run_ratio(0.3, root, meta, args, torch.device("cpu"))
                args.out = str(root / "resumed")

                def interrupt(path, value):
                    atomic_checkpoint(path, value)
                    raise InterruptedError("simulated interruption after epoch checkpoint")

                with patch("train_ista.atomic_checkpoint", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        run_ratio(0.3, root, meta, args, torch.device("cpu"))
                args.resume = True
                resumed = run_ratio(0.3, root, meta, args, torch.device("cpu"))
            self.assertEqual(complete["loss_hist"], resumed["loss_hist"])
            self.assertEqual(complete["metrics"], resumed["metrics"])
            checkpoint = torch.load(root/"resumed"/resumed["checkpoint"], weights_only=True)
            net = ISTANetPlus(1, 2)
            net.load_state_dict(checkpoint["model"])
            ds = ReconSet(root, ["test"])
            with patch.object(net, "forward", wraps=net.forward) as forward:
                result = evaluate(net, DataLoader(ds, batch_size=2),
                                  checkpoint["hard_mask"], "cpu", ds.case_of, False)
                self.assertTrue(all(c.kwargs["final_hard_dc"] is False
                                    for c in forward.call_args_list))
            self.assertAlmostEqual(result["psnr"], resumed["metrics"]["psnr"], places=5)
            args.epochs = 3
            with self.assertRaises(ValueError):
                run_ratio(0.3, root, meta, args, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
