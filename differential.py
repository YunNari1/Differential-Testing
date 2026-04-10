from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class DifferentialResult:
    index: int
    true_label: int
    preds_before: List[int]
    preds_after: List[int]
    found_disagreement_before: bool
    found_disagreement_after: bool
    coverage_before: float
    coverage_after: float
    adv_image: torch.Tensor


def predictions_from_models(models, x):
    preds = []
    logits_all = []
    for model in models:
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)
        logits_all.append(logits)
    return preds, logits_all


def has_disagreement(preds: List[int]) -> bool:
    return len(set(preds)) > 1


def generate_disagreement_input(models, x, y, eps=0.03, alpha=0.007, steps=10):
    x_adv = x.clone().detach()
    x_adv = x_adv + 0.001 * torch.randn_like(x_adv)
    x_adv = x_adv.clamp(0.0, 1.0)
    x_orig = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)

        logits_list = [m(x_adv) for m in models]
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]

        disagreement_loss = 0.0
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                p = probs_list[i]
                q = probs_list[j]
                disagreement_loss += F.kl_div(q.log(), p, reduction="batchmean")
                disagreement_loss += F.kl_div(p.log(), q, reduction="batchmean")

        miscls_loss = 0.0
        for logits in logits_list:
            miscls_loss += -F.cross_entropy(logits, y)

        loss = disagreement_loss + miscls_loss

        for m in models:
            m.zero_grad(set_to_none=True)

        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        delta = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

    return x_adv


def run_differential_test_on_batch(
    models,
    x,
    y,
    batch_start_idx,
    coverage_trackers=None,
    eps=0.03,
    alpha=0.007,
    steps=10,
):
    results = []
    batch_size = x.shape[0]

    for i in range(batch_size):
        xi = x[i:i+1]
        yi = y[i:i+1]

        preds_before, _ = predictions_from_models(models, xi)
        before_disagreement = has_disagreement(preds_before)

        cov_before = 0.0
        if coverage_trackers is not None:
            ratios = []
            for tracker, model in zip(coverage_trackers, models):
                _ = model(xi)
                _, _, ratio = tracker.coverage()
                ratios.append(ratio)
            cov_before = sum(ratios) / len(ratios)

        x_adv = generate_disagreement_input(
            models=models,
            x=xi,
            y=yi,
            eps=eps,
            alpha=alpha,
            steps=steps,
        )

        preds_after, _ = predictions_from_models(models, x_adv)
        after_disagreement = has_disagreement(preds_after)

        cov_after = cov_before
        if coverage_trackers is not None:
            ratios = []
            for tracker, model in zip(coverage_trackers, models):
                _ = model(x_adv)
                _, _, ratio = tracker.coverage()
                ratios.append(ratio)
            cov_after = sum(ratios) / len(ratios)

        results.append(
            DifferentialResult(
                index=batch_start_idx + i,
                true_label=yi.item(),
                preds_before=preds_before,
                preds_after=preds_after,
                found_disagreement_before=before_disagreement,
                found_disagreement_after=after_disagreement,
                coverage_before=cov_before,
                coverage_after=cov_after,
                adv_image=x_adv.detach().cpu(),
            )
        )

    return results