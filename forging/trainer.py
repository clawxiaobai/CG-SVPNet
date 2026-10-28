# Extracted scientific kernel; provenance: reference/source_provenance.json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv, json, time
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import load_case_assets_mpm
from .model import CoinEquivalentStateModel, REQUESTED_MODES
from .protocol import (seed_all, make_state_dataset, move_state_batch, batch_one,
    STATE_FIELDS, STATE_NAMES, _new_accumulator, _update_accumulator, _finish_accumulator)
from . import hooks
def multitask_loss(output, sample, args):
    disp_rmse = torch.sqrt(
        F.mse_loss(output["disp"], sample["target_disp"]) + 1.0e-12
    )
    state_mse = (
        output["state"] - sample["target_state"]
    ).square().mean(dim=(0, 1))
    state_rmse = torch.sqrt(state_mse + 1.0e-12)
    late_factor = 1.0 + args.late_loss_weight * sample[
        "progress"
    ].mean().pow(8)
    total = late_factor * (
        args.disp_weight * disp_rmse
        + args.strain_weight * state_rmse[0]
        + args.stress_weight * state_rmse[1]
    )
    return total, disp_rmse.detach(), state_rmse.detach()


def _selection_score(metrics, args):
    def state_score(prefix):
        return 0.5 * (
            0.2 * metrics[f"{prefix}equiv_strain_rmse"]
            + 0.02 * metrics[f"{prefix}effective_stress_rmse"]
        )

    all_score = metrics["disp_rmse"] + (
        args.state_select_weight * state_score("")
    )
    late_score = metrics.get("late_disp_rmse", metrics["disp_rmse"]) + (
        args.state_select_weight * state_score("late_")
    )
    final_score = metrics.get("final_disp_rmse", late_score) + (
        args.state_select_weight * state_score("final_")
    )
    return (
        all_score
        + args.select_late_weight * late_score
        + args.select_final_weight * final_score
    )


@torch.no_grad()
def evaluate(model, dataset, assets, device, include_cases=False):
    model.eval()
    pools = {
        "": _new_accumulator(),
        "late_": _new_accumulator(),
        "final_": _new_accumulator(),
    }
    case_pools = {}
    for index in range(len(dataset)):
        hooks.check_stop()
        hooks.event("evaluate", step=index+1, total=len(dataset))
        sample = move_state_batch(batch_one(dataset[index]), device)
        output = model(sample, assets)
        scale = sample["target_state_scale"][:, None, :]
        pred_raw = output["state"].float() / scale
        target_raw = sample["target_state_raw"]
        half_scale = sample["scale"][:, None, :] * 0.5
        disp_diff_world = (
            output["disp"].float() - sample["target_disp"]
        ) * half_scale
        _update_accumulator(
            pools[""], pred_raw, target_raw, disp_diff_world
        )
        progress = float(sample["progress"].item())
        if progress >= 0.70:
            _update_accumulator(
                pools["late_"], pred_raw, target_raw, disp_diff_world
            )
        if progress >= 0.999:
            _update_accumulator(
                pools["final_"], pred_raw, target_raw, disp_diff_world
            )

        if include_cases:
            case = sample["case"]
            if case not in case_pools:
                case_pools[case] = {
                    "": _new_accumulator(),
                    "late_": _new_accumulator(),
                    "final_": _new_accumulator(),
                }
            _update_accumulator(
                case_pools[case][""], pred_raw, target_raw,
                disp_diff_world,
            )
            if progress >= 0.70:
                _update_accumulator(
                    case_pools[case]["late_"], pred_raw, target_raw,
                    disp_diff_world,
                )
            if progress >= 0.999:
                _update_accumulator(
                    case_pools[case]["final_"], pred_raw, target_raw,
                    disp_diff_world,
                )

    metrics = {}
    for prefix, accumulator in pools.items():
        if accumulator["count"]:
            metrics.update(_finish_accumulator(accumulator, prefix))
    metrics["selection_score"] = _selection_score(metrics, evaluate.args)

    cases = {}
    for case, prefixed in case_pools.items():
        case_metrics = {}
        for prefix, accumulator in prefixed.items():
            if accumulator["count"]:
                case_metrics.update(
                    _finish_accumulator(accumulator, prefix)
                )
        cases[case] = case_metrics
    return metrics, cases


def train(args):
    seed_all(args.seed)
    evaluate.args = args
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment_dir = Path(args.output_root) / args.exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = make_state_dataset("train", args, train=True)
    val_dataset = make_state_dataset("val", args, train=False)
    test_dataset = make_state_dataset("test", args, train=False)
    if args.profile == "smoke":
        train_dataset.samples = train_dataset.samples[:2]
        val_dataset.samples = [max(val_dataset.samples, key=lambda s: (s['target_progress_pos'], s['case_name']))]
        test_dataset.samples = [max(test_dataset.samples, key=lambda s: (s['target_progress_pos'], s['case_name']))]
    assets = load_case_assets_mpm(
        str(args.data_root), device, asset_variant="contact"
    )
    model = CoinEquivalentStateModel(
        res=args.res,
        width=args.width,
        decoder_mode=args.decoder_mode,
        hidden_dim=args.hidden_dim,
        point_chunk=args.point_chunk,
        mold_input_mode=args.mold_input_mode,
        friction_coefficient=args.friction_coefficient,
        state_init=args.state_init,
        residual_gate_mode=args.residual_gate_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    start_epoch = 1
    best_score = math.inf
    stale = 0
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None:
        checkpoint = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        resume_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = resume_epoch + 1
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Older checkpoints did not store optimizer/scheduler state.  Keep
            # the completed cosine schedule position instead of restarting it.
            eta_min = args.lr * 0.05
            lr = eta_min + 0.5 * (args.lr - eta_min) * (
                1.0 + math.cos(math.pi * resume_epoch / max(args.epochs, 1))
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
                group.setdefault("initial_lr", args.lr)
            scheduler.last_epoch = resume_epoch
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            scheduler._step_count = resume_epoch + 1
        best_path = experiment_dir / "best_model.pth"
        if best_path.exists():
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_score = float(
                best_checkpoint.get("metrics", {}).get(
                    "selection_score", math.inf
                )
            )
            best_epoch = int(best_checkpoint.get("epoch", resume_epoch))
            stale = int(checkpoint.get(
                "stale", max(0, resume_epoch - best_epoch)
            ))
        else:
            best_score = float(checkpoint.get(
                "best_score",
                checkpoint.get("metrics", {}).get("selection_score", math.inf),
            ))
            stale = int(checkpoint.get("stale", 0))
        print(
            f"resuming from {resume_path} at epoch {start_epoch}; "
            f"lr={optimizer.param_groups[0]['lr']:.8g} "
            f"best={best_score:.6f} stale={stale}",
            flush=True,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update({
        "model_label": REQUESTED_MODES[args.decoder_mode],
        "state_fields": STATE_FIELDS,
        "state_names": STATE_NAMES,
        "state_scales": {"effstn": 0.2, "effsts": 0.02},
        "target_state_is_input": False,
        "training_origin": (
            f"resumed_from:{resume_path}" if resume_path is not None
            else "random_initialization"
        ),
        "protocol": "first_valid_frame_to_any_later_frame",
        "train_cases": len({s["case_name"] for s in train_dataset.samples}),
        "train_states": len(train_dataset),
        "val_states": len(val_dataset),
        "test_states": len(test_dataset),
    })
    (experiment_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    log_fields = [
        "epoch", "lr", "train_total", "train_disp_norm_rmse",
        "train_strain_norm_rmse", "train_stress_norm_rmse",
        "equiv_strain_rmse", "effective_stress_rmse", "disp_rmse",
        "late_equiv_strain_rmse", "late_effective_stress_rmse",
        "late_disp_rmse", "final_equiv_strain_rmse",
        "final_effective_stress_rmse", "final_disp_rmse",
        "selection_score", "seconds",
    ]
    log_path = experiment_dir / "train_log.csv"
    append_log = resume_path is not None and log_path.exists()
    with log_path.open(
        "a" if append_log else "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=log_fields)
        if not append_log or log_path.stat().st_size == 0:
            writer.writeheader()
        print(
            f"{args.exp_name} | {REQUESTED_MODES[args.decoder_mode]} | "
            f"device={device} train={len(train_dataset)} "
            f"val={len(val_dataset)} test={len(test_dataset)} "
            f"params={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
        for epoch in range(start_epoch, args.epochs + 1):
            hooks.check_stop()
            hooks.event("train", epoch=epoch, epochs=args.epochs)
            model.train()
            train_dataset.sampling_seed_offset = (
                args.seed + epoch * 1000003
            )
            totals = []
            disps = []
            strains = []
            stresses = []
            started = time.time()
            batch_limit = (
                len(train_dataset)
                if args.samples_per_epoch <= 0
                else min(len(train_dataset), args.samples_per_epoch)
            )
            for batch_index, batch in enumerate(loader, 1):
                if batch_index > batch_limit:
                    break
                hooks.check_stop()
                sample = move_state_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model(sample, assets)
                    total, disp_rmse, state_rmse = multitask_loss(
                        output, sample, args
                    )
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                hooks.event("train", epoch=epoch, epochs=args.epochs, batch=batch_index, batches=batch_limit, loss=float(total.detach().cpu()))
                totals.append(float(total.detach().cpu()))
                disps.append(float(disp_rmse.cpu()))
                strains.append(float(state_rmse[0].cpu()))
                stresses.append(float(state_rmse[1].cpu()))
                if batch_index % args.print_every == 0:
                    print(
                        f"E{epoch:03d} batch={batch_index}/{batch_limit} "
                        f"loss={totals[-1]:.5f} "
                        f"disp={disps[-1]:.5f} "
                        f"strain={strains[-1]:.5f} "
                        f"stress={stresses[-1]:.5f}",
                        flush=True,
                    )
            scheduler.step()

            if epoch != 1 and epoch % args.val_every != 0:
                print(
                    f"E{epoch:03d} train={np.mean(totals):.5f} "
                    "validation=deferred",
                    flush=True,
                )
                continue

            metrics, _ = evaluate(
                model, val_dataset, assets, device,
                include_cases=False,
            )
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_total": float(np.mean(totals)),
                "train_disp_norm_rmse": float(np.mean(disps)),
                "train_strain_norm_rmse": float(np.mean(strains)),
                "train_stress_norm_rmse": float(np.mean(stresses)),
                **{key: metrics.get(key, math.nan) for key in log_fields
                   if key in metrics},
                "seconds": time.time() - started,
            }
            writer.writerow(row)
            stream.flush()
            print(
                f"E{epoch:03d} val disp={metrics['disp_rmse']:.4f}mm "
                f"strain={metrics['equiv_strain_rmse']:.4f} "
                f"stress={metrics['effective_stress_rmse']:.4f} "
                f"score={metrics['selection_score']:.5f}",
                flush=True,
            )
            improved = metrics["selection_score"] < best_score
            if improved:
                best_score = metrics["selection_score"]
                stale = 0
            else:
                stale += args.val_every
            checkpoint = {
                "epoch": epoch,
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_score": best_score,
                "stale": stale,
                "args": config,
                "task": "coin_equivalent_state_multimodel",
            }
            torch.save(
                checkpoint, experiment_dir / "checkpoint_latest.pth"
            )
            if improved:
                torch.save(checkpoint, experiment_dir / "best_model.pth")
                print(
                    f"E{epoch:03d} new best score={best_score:.6f}",
                    flush=True,
                )
            if stale >= args.patience:
                print(
                    f"early stop at epoch {epoch}; "
                    f"no improvement for {stale} epochs",
                    flush=True,
                )
                break

    best = torch.load(
        experiment_dir / "best_model.pth",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"], strict=True)
    validation, validation_cases = evaluate(
        model, val_dataset, assets, device, include_cases=True
    )
    test, test_cases = evaluate(
        model, test_dataset, assets, device, include_cases=True
    )
    final = {
        "best_epoch": int(best["epoch"]),
        "residual_gates": (
            {
                "displacement": float(
                    torch.sigmoid(model.residual_gain_disp).detach().cpu()
                ),
                "equivalent_strain": float(
                    torch.sigmoid(model.residual_gain_strain).detach().cpu()
                ),
                "effective_stress": float(
                    torch.sigmoid(model.residual_gain_stress).detach().cpu()
                ),
            }
            if args.residual_gate_mode == "three"
            else (
                {
                    "shared": float(
                        torch.sigmoid(model.residual_gain).detach().cpu()
                    )
                }
                if hasattr(model, "residual_gain") else {}
            )
        ),
        "validation": validation,
        "validation_by_case": validation_cases,
        "test": test,
        "test_by_case": test_cases,
    }
    (experiment_dir / "final_metrics.json").write_text(
        json.dumps(final, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(final, indent=2, allow_nan=True), flush=True)
    return final

