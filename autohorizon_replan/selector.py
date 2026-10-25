"""AutoHorizon's published implementation, followed by an optional execution cap.

Reference: ``bidir_soft_pointer`` and ``_soft_pointer_prefix`` in
https://github.com/hatchetProject/AutoHorizon/blob/
c7504f1756109103f2cfcc2e23f1b1a23841c885/src/openpi/models_pytorch/pi0_pytorch.py
(lines 625--723). We preserve its mapped-original-index backward ``N_b`` and
its additional pointer-join test, rather than substituting the paper's simpler
coverage description. The plateau condition is a SMALL pointer increment.

The caller supplies the action-to-action matrix averaged across layers/heads
at FM evaluation 3 (one-based). There is no averaging across FM evaluations.
All selection arithmetic runs in torch.float32 on CPU so a saved float32
matrix can reproduce the decision independently of the policy's GPU device.
This deliberate device choice may differ from GPU reduction rounding exactly
at a threshold. Tensor operations and the two row-normalization passes match
the reference implementation. H=1 is explicitly defined; the reference's
normalized entropy divides by log(1) and does not cover this edge case.
"""

import copy
import math

UPSTREAM_COMMIT = "c7504f1756109103f2cfcc2e23f1b1a23841c885"
UPSTREAM_URL = (
    f"https://github.com/hatchetProject/AutoHorizon/blob/{UPSTREAM_COMMIT}"
    "/src/openpi/models_pytorch/pi0_pytorch.py"
)
SELECTOR_VARIANT = "official_code_mapped_backward_join_v1"
_EPS = 1e-12


def _prefix(matrix, hold_thr, max_entropy_q, torch):
    """One direction of the reference rule, with all diagnostic intermediates."""
    horizon = matrix.shape[0]
    matrix = matrix / (matrix.sum(-1, keepdim=True) + _EPS)
    indices = torch.arange(horizon, dtype=matrix.dtype, device=matrix.device)
    expected_index = (matrix * indices).sum(-1)
    pointer = torch.maximum(expected_index, torch.cummax(expected_index, dim=0).values)
    probabilities = matrix.clamp_min(_EPS)
    entropy = -(probabilities * probabilities.log()).sum(-1) / math.log(horizon)
    entropy_threshold = torch.quantile(entropy.float(), q=min(max_entropy_q, 0.999))
    reliable = entropy <= entropy_threshold
    previous = torch.cat([pointer.new_tensor([0.0]), pointer[:-1]])
    increments = pointer - previous
    hold = (increments < hold_thr) & reliable
    hold_indices = torch.nonzero(hold, as_tuple=False)
    stop = int(hold_indices[0].item()) if len(hold_indices) else horizon - 1
    return {
        "expected_index": expected_index,
        "pointer": pointer,
        "increments": increments,
        "entropy": entropy,
        "entropy_threshold": entropy_threshold,
        "reliable": reliable,
        "hold": hold,
        "stop": stop,
        "found_hold": bool(len(hold_indices)),
    }


class AutoHorizonSelector:
    """Stateless attention selector with a cap applied after the full decision.

    ``select`` accepts a tensor, NumPy array, or nested numeric sequence of
    shape ``(horizon, horizon)``. Attention values must be finite/nonnegative
    and every row must have positive mass. It returns JSON-compatible values;
    fields shared with the existing executor intentionally carry no DVAC
    history/threshold semantics. ``last_decision`` is an independent copy.
    """

    def __init__(self, horizon=50, execution_cap=None, *, hold_thr=0.3,
                 max_entropy_q=0.9, selected_fm_step=3):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if execution_cap is None:
            execution_cap = horizon
        if type(execution_cap) is not int or not 1 <= execution_cap <= horizon:
            raise ValueError("execution_cap must be an integer in [1, horizon]")
        if isinstance(hold_thr, bool) or not isinstance(hold_thr, (int, float)):
            raise ValueError("hold_thr must be a finite nonnegative number")
        if not math.isfinite(hold_thr) or hold_thr < 0:
            raise ValueError("hold_thr must be a finite nonnegative number")
        if isinstance(max_entropy_q, bool) or not isinstance(max_entropy_q, (int, float)):
            raise ValueError("max_entropy_q must be in (0, 1]")
        if not math.isfinite(max_entropy_q) or not 0 < max_entropy_q <= 1:
            raise ValueError("max_entropy_q must be in (0, 1]")
        if type(selected_fm_step) is not int or selected_fm_step < 1:
            raise ValueError("selected_fm_step must be a positive one-based integer")
        self.horizon = horizon
        self.execution_cap = execution_cap
        self.hold_thr = float(hold_thr)
        self.max_entropy_q = float(max_entropy_q)
        self.selected_fm_step = selected_fm_step
        self.last_decision = None

    def reset(self):
        self.last_decision = None

    def select(self, attention):
        import torch

        try:
            source = torch.as_tensor(attention)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("attention must be a rectangular numeric matrix") from exc
        if source.is_complex() or source.dtype == torch.bool:
            raise ValueError("attention must contain real numeric values, not complex or bool")
        # Validate before float32 conversion as well, so tiny negative values
        # cannot silently become -0.0 and nonfinite source values cannot hide.
        if isinstance(attention, torch.Tensor):
            validation_values = attention.detach().to(device="cpu", dtype=torch.float64)
        else:
            validation_values = torch.as_tensor(attention, device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(validation_values).all()) or bool((validation_values < 0).any()):
            raise ValueError("attention values must be finite and nonnegative")
        matrix = source.detach().to(device="cpu", dtype=torch.float32).clone()
        if matrix.ndim != 2 or tuple(matrix.shape) != (self.horizon, self.horizon):
            raise ValueError(f"attention must have shape ({self.horizon}, {self.horizon})")
        if not bool(torch.isfinite(matrix).all()) or bool((matrix < 0).any()):
            raise ValueError("attention values must be finite and nonnegative in float32")
        row_mass = matrix.sum(-1, keepdim=True)
        if not bool(torch.isfinite(row_mass).all()) or bool((row_mass <= 0).any()):
            raise ValueError("every attention row must have finite, strictly positive mass")
        normalized = matrix / (row_mass + _EPS)
        if self.horizon == 1:
            diagnostics = self._single_action_diagnostics(normalized, self.hold_thr > 0)
            raw = 1
        else:
            forward = _prefix(normalized, self.hold_thr, self.max_entropy_q, torch)
            reversed_matrix = torch.flip(normalized, dims=(0, 1))
            backward = _prefix(reversed_matrix, self.hold_thr, self.max_entropy_q, torch)
            backward_pointer = torch.flip(self.horizon - 1 - backward["pointer"], dims=(0,))
            forward_count = int(torch.clamp(
                torch.floor(forward["pointer"][forward["stop"]]) + 1, 1, self.horizon).item())
            backward_stop_original = self.horizon - 1 - backward["stop"]
            # This is intentionally the upstream mapped-index definition, not
            # floor(backward["pointer"][backward["stop"]]) + 1.
            backward_count = int(torch.clamp(
                torch.floor(backward_pointer[backward_stop_original]) + 1, 1, self.horizon).item())
            gap = backward_pointer - forward["pointer"]
            joins = torch.nonzero(gap <= 1.0, as_tuple=False)
            join_row = int(joins[0].item()) if len(joins) else None
            full_coverage = forward_count + backward_count >= self.horizon and join_row is not None
            raw = self.horizon if full_coverage else forward_count
            diagnostics = {
                "normalized_attention": normalized.tolist(),
                "mu_forward": forward["pointer"].tolist(),
                "expected_index_forward": forward["expected_index"].tolist(),
                "dmu_forward": forward["increments"].tolist(),
                "entropy_forward": forward["entropy"].tolist(),
                "entropy_threshold_forward": float(forward["entropy_threshold"].item()),
                "reliable_forward": forward["reliable"].tolist(),
                "holds_forward": forward["hold"].tolist(),
                "stop_forward": forward["stop"],
                "found_hold_forward": forward["found_hold"],
                "N_forward": forward_count,
                "mu_backward": backward_pointer.tolist(),
                "mu_backward_reversed": backward["pointer"].tolist(),
                "expected_index_backward_reversed": backward["expected_index"].tolist(),
                "dmu_backward_reversed": backward["increments"].tolist(),
                "entropy_backward_reversed": backward["entropy"].tolist(),
                "entropy_threshold_backward": float(backward["entropy_threshold"].item()),
                "reliable_backward_reversed": backward["reliable"].tolist(),
                "holds_backward_reversed": backward["hold"].tolist(),
                "stop_backward_reversed": backward["stop"],
                "stop_backward": backward_stop_original,
                "found_hold_backward": backward["found_hold"],
                "N_backward": backward_count,
                "join_row": join_row,
                "gap": gap.tolist(),
                "full_coverage": full_coverage,
                "single_action_extension": False,
            }
        decision = {
            "chosen_k": min(raw, self.execution_cap),
            "raw_chosen_k": raw,
            "execution_cap": self.execution_cap,
            "cap_applied": raw > self.execution_cap,
            "threshold": None,
            "first_crossing": None,
            "window_calls": 0,
            "window_mean": None,
            "window_std": None,
            "bootstrap": False,
            "method": "autohorizon",
            "method_id": f"autohorizon_cap_{self.execution_cap}",
            "selector_variant": SELECTOR_VARIANT,
            "upstream_commit": UPSTREAM_COMMIT,
            "selected_fm_step": self.selected_fm_step,
            "hold_thr": self.hold_thr,
            "max_entropy_q": self.max_entropy_q,
            "effective_entropy_q": min(self.max_entropy_q, 0.999),
            "run_len": 1,
            "selector_device": "cpu",
            "selector_dtype": "float32",
            "diagnostics": diagnostics,
        }
        self.last_decision = copy.deepcopy(decision)
        return decision

    @staticmethod
    def _single_action_diagnostics(normalized, holds):
        return {
            "normalized_attention": normalized.tolist(),
            "mu_forward": [0.0], "expected_index_forward": [0.0], "dmu_forward": [0.0],
            "entropy_forward": [0.0], "entropy_threshold_forward": 0.0,
            "reliable_forward": [True], "holds_forward": [holds],
            "stop_forward": 0, "found_hold_forward": holds, "N_forward": 1,
            "mu_backward": [0.0], "mu_backward_reversed": [0.0],
            "expected_index_backward_reversed": [0.0], "dmu_backward_reversed": [0.0],
            "entropy_backward_reversed": [0.0], "entropy_threshold_backward": 0.0,
            "reliable_backward_reversed": [True], "holds_backward_reversed": [holds],
            "stop_backward_reversed": 0, "stop_backward": 0,
            "found_hold_backward": holds, "N_backward": 1, "join_row": 0,
            "gap": [0.0], "full_coverage": True, "single_action_extension": True,
        }
