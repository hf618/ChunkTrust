import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

FEATURE_NAMES = [
    "vnorm_late",
    "align_late",
    "rperp_end",
    "kappa_end",
    "eHH_late",
    "eHL_late",
    "eLH_late",
    "eLL_late",
]


def _episodes_dir(run_dir: Path) -> Path:
    d = run_dir / "episodes"
    return d if d.exists() else run_dir


def _load_xt_internal_or_xt(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "XT_internal" in data.files:
        return np.asarray(data["XT_internal"], dtype=np.float64)
    return np.asarray(data["XT"], dtype=np.float64)


def _spectrum_rfft2(mat: np.ndarray) -> np.ndarray:
    vv = np.fft.rfftn(mat, axes=(0, 1))
    return (vv.real * vv.real) + (vv.imag * vv.imag)


def _split_quadrants(spec: np.ndarray, cut_t: float, cut_a: float, eps: float) -> dict[str, float]:
    f_t = spec.shape[0]
    f_a = spec.shape[1]
    ft_c = max(1, int(cut_t * f_t))
    fa_c = max(1, int(cut_a * f_a))
    e_ll = float(np.sum(spec[:ft_c, :fa_c]))
    e_lh = float(np.sum(spec[:ft_c, fa_c:]))
    e_hl = float(np.sum(spec[ft_c:, :fa_c]))
    e_hh = float(np.sum(spec[ft_c:, fa_c:]))
    e_all = e_ll + e_lh + e_hl + e_hh
    denom = e_all + eps
    return {
        "eLL": e_ll / denom,
        "eLH": e_lh / denom,
        "eHL": e_hl / denom,
        "eHH": e_hh / denom,
    }


def compute_step_features(
    v_step: np.ndarray,
    ds_step: np.ndarray,
    x0_step: np.ndarray,
    xt_step: np.ndarray,
    *,
    late_tau_start: int = 7,
    cut_t: float = 0.25,
    cut_a: float = 0.25,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    n_tau, h, d_a = v_step.shape
    d_flat = h * d_a
    lo_late = max(0, min(n_tau - 1, int(late_tau_start)))

    v_flat = v_step.reshape(n_tau, d_flat)
    dx = ds_step[:, None] * v_flat
    vnorm = np.linalg.norm(v_flat, axis=-1)
    vnorm_late = float(np.mean(vnorm[lo_late:]))

    c = (xt_step - x0_step).reshape(d_flat)
    p = np.cumsum(dx, axis=0)
    dot_pc = np.sum(p * c[None, :], axis=-1)
    norm_p = np.linalg.norm(p, axis=-1)
    norm_c = float(np.linalg.norm(c))
    align = dot_pc / (norm_p * norm_c + eps)
    align_late = float(np.mean(align[lo_late:]))

    dx_norm = np.linalg.norm(dx, axis=-1)
    l_prefix = np.cumsum(dx_norm, axis=0)
    p_norm = norm_p
    kappa = l_prefix / (p_norm + eps)
    kappa_end = float(kappa[-1])

    d_unit = c / (norm_c + eps)
    proj = np.sum(dx * d_unit[None, :], axis=-1, keepdims=True) * d_unit[None, :]
    dx_perp = dx - proj
    eperp = np.cumsum(np.sum(dx_perp * dx_perp, axis=-1), axis=0)
    eall = np.cumsum(np.sum(dx * dx, axis=-1), axis=0)
    rperp = eperp / (eall + eps)
    rperp_end = float(rperp[-1])

    quad_rows = []
    for tau in range(n_tau):
        spec = _spectrum_rfft2(v_step[tau])
        quad_rows.append(_split_quadrants(spec, cut_t=cut_t, cut_a=cut_a, eps=eps))
    ehh_late = float(np.mean([q["eHH"] for q in quad_rows[lo_late:]]))
    ehl_late = float(np.mean([q["eHL"] for q in quad_rows[lo_late:]]))
    elh_late = float(np.mean([q["eLH"] for q in quad_rows[lo_late:]]))
    ell_late = float(np.mean([q["eLL"] for q in quad_rows[lo_late:]]))

    feats = {
        "vnorm_late": vnorm_late,
        "align_late": align_late,
        "rperp_end": rperp_end,
        "kappa_end": kappa_end,
        "eHH_late": ehh_late,
        "eHL_late": ehl_late,
        "eLH_late": elh_late,
        "eLL_late": ell_late,
    }
    vec = np.asarray([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
    return vec, feats


def compute_step_features_jax(
    v_step: jnp.ndarray,
    ds_step: jnp.ndarray,
    x0_step: jnp.ndarray,
    xt_step: jnp.ndarray,
    *,
    late_tau_start: int = 7,
    cut_t: float = 0.25,
    cut_a: float = 0.25,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """JAX implementation of online step features for inference-time chunk control."""
    v_step = jnp.asarray(v_step, dtype=jnp.float32)
    ds_step = jnp.asarray(ds_step, dtype=jnp.float32)
    x0_step = jnp.asarray(x0_step, dtype=jnp.float32)
    xt_step = jnp.asarray(xt_step, dtype=jnp.float32)

    n_tau, h, d_a = v_step.shape
    lo_late = jnp.clip(jnp.asarray(late_tau_start, dtype=jnp.int32), 0, n_tau - 1)
    d_flat = h * d_a

    v_flat = jnp.reshape(v_step, (n_tau, d_flat))
    dx = ds_step[:, None] * v_flat
    vnorm = jnp.linalg.norm(v_flat, axis=-1)
    vnorm_late = jnp.mean(vnorm[lo_late:])

    c = jnp.reshape(xt_step - x0_step, (d_flat,))
    p = jnp.cumsum(dx, axis=0)
    dot_pc = jnp.sum(p * c[None, :], axis=-1)
    norm_p = jnp.linalg.norm(p, axis=-1)
    norm_c = jnp.linalg.norm(c)
    align = dot_pc / (norm_p * norm_c + eps)
    align_late = jnp.mean(align[lo_late:])

    dx_norm = jnp.linalg.norm(dx, axis=-1)
    l_prefix = jnp.cumsum(dx_norm, axis=0)
    kappa = l_prefix / (norm_p + eps)
    kappa_end = kappa[-1]

    d_unit = c / (norm_c + eps)
    proj = jnp.sum(dx * d_unit[None, :], axis=-1, keepdims=True) * d_unit[None, :]
    dx_perp = dx - proj
    eperp = jnp.cumsum(jnp.sum(dx_perp * dx_perp, axis=-1), axis=0)
    eall = jnp.cumsum(jnp.sum(dx * dx, axis=-1), axis=0)
    rperp = eperp / (eall + eps)
    rperp_end = rperp[-1]

    def _quad_one_tau(v_tau: jnp.ndarray) -> jnp.ndarray:
        spec = jnp.fft.rfftn(v_tau, axes=(0, 1))
        pwr = (jnp.real(spec) * jnp.real(spec)) + (jnp.imag(spec) * jnp.imag(spec))
        f_t, f_a = pwr.shape[0], pwr.shape[1]
        ft_c = jnp.maximum(1, jnp.asarray(cut_t * f_t, dtype=jnp.int32))
        fa_c = jnp.maximum(1, jnp.asarray(cut_a * f_a, dtype=jnp.int32))
        e_ll = jnp.sum(pwr[:ft_c, :fa_c])
        e_lh = jnp.sum(pwr[:ft_c, fa_c:])
        e_hl = jnp.sum(pwr[ft_c:, :fa_c])
        e_hh = jnp.sum(pwr[ft_c:, fa_c:])
        e_all = e_ll + e_lh + e_hl + e_hh + eps
        return jnp.stack([e_hh / e_all, e_hl / e_all, e_lh / e_all, e_ll / e_all], axis=0)

    quads = jnp.stack([_quad_one_tau(v_step[tau]) for tau in range(n_tau)], axis=0)  # (N_tau, 4)
    quad_late = quads[lo_late:, :]
    ehh_late = jnp.mean(quad_late[:, 0])
    ehl_late = jnp.mean(quad_late[:, 1])
    elh_late = jnp.mean(quad_late[:, 2])
    ell_late = jnp.mean(quad_late[:, 3])

    return jnp.stack(
        [vnorm_late, align_late, rperp_end, kappa_end, ehh_late, ehl_late, elh_late, ell_late], axis=0
    )


def _ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows: list[dict[str, float]] = []
    n = y_true.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            rows.append({"bin_left": float(lo), "bin_right": float(hi), "count": 0, "conf": float("nan"), "acc": float("nan")})
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        frac = float(np.sum(mask) / n)
        ece += abs(conf - acc) * frac
        rows.append({"bin_left": float(lo), "bin_right": float(hi), "count": int(np.sum(mask)), "conf": conf, "acc": acc})
    return float(ece), rows


def fit_prior_from_default_run(
    base_default_run_dir: Path,
    *,
    force_refit: bool = False,
    late_tau_start: int = 7,
    cut_t: float = 0.25,
    cut_a: float = 0.25,
    eps: float = 1e-12,
    random_seed: int = 0,
) -> dict[str, Any]:
    prior_dir = base_default_run_dir / "prior"
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_model_path = prior_dir / "prior_model.json"
    prior_report_path = prior_dir / "prior_report.json"
    prior_pred_path = prior_dir / "prior_predictions.csv"

    if prior_model_path.exists() and not force_refit:
        return load_prior_model(prior_model_path)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError(f"scikit-learn unavailable for prior fitting: {e}") from e

    xs: list[np.ndarray] = []
    ys: list[int] = []
    ep_ids: list[int] = []
    for npz_path in sorted(_episodes_dir(base_default_run_dir).glob("episode*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            required = {"v", "ds", "X0", "success"}
            if (not required.issubset(set(data.files))) or ("XT" not in data.files and "XT_internal" not in data.files):
                continue
            v = np.asarray(data["v"], dtype=np.float64)
            ds = np.asarray(data["ds"], dtype=np.float64)
            x0 = np.asarray(data["X0"], dtype=np.float64)
            xt = _load_xt_internal_or_xt(data)
            success = int(bool(np.asarray(data["success"]).item()))
            if v.ndim != 4:
                continue
            t_steps = v.shape[0]
            if t_steps == 0:
                continue
            per_t = []
            for t in range(t_steps):
                vec, _ = compute_step_features(
                    v[t],
                    ds[t],
                    x0[t],
                    xt[t],
                    late_tau_start=late_tau_start,
                    cut_t=cut_t,
                    cut_a=cut_a,
                    eps=eps,
                )
                per_t.append(vec)
            x_ep = np.mean(np.stack(per_t, axis=0), axis=0)
            xs.append(x_ep)
            ys.append(success)
            ep_ids.append(int(npz_path.stem.replace("episode", "")))

    if not xs:
        raise RuntimeError(f"No usable episode*.npz for prior fit under {base_default_run_dir}")

    x = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        raise RuntimeError("Not enough positive/negative samples to fit CV prior.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    oof = np.zeros_like(y, dtype=np.float64)
    auc_list: list[float] = []
    pr_list: list[float] = []
    brier_list: list[float] = []
    for tr, te in cv.split(x, y):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x[tr])
        xte = scaler.transform(x[te])
        model = LogisticRegression(max_iter=3000, random_state=random_seed)
        model.fit(xtr, y[tr])
        prob = model.predict_proba(xte)[:, 1]
        oof[te] = prob
        auc_list.append(float(roc_auc_score(y[te], prob)))
        pr_list.append(float(average_precision_score(y[te], prob)))
        brier_list.append(float(brier_score_loss(y[te], prob)))

    scaler_full = StandardScaler()
    x_full = scaler_full.fit_transform(x)
    model_full = LogisticRegression(max_iter=3000, random_state=random_seed)
    model_full.fit(x_full, y)

    ece, reliability = _ece_score(y, oof, n_bins=10)
    report = {
        "n_samples": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "cv_n_splits": int(n_splits),
        "auc_mean": float(np.mean(auc_list)),
        "auc_std": float(np.std(auc_list)),
        "pr_auc_mean": float(np.mean(pr_list)),
        "brier_mean": float(np.mean(brier_list)),
        "auc_oof": float(roc_auc_score(y, oof)),
        "pr_auc_oof": float(average_precision_score(y, oof)),
        "brier_oof": float(brier_score_loss(y, oof)),
        "ece_oof": float(ece),
        "reliability_bins": reliability,
        "feature_names": FEATURE_NAMES,
        "fit_params": {
            "late_tau_start": int(late_tau_start),
            "cut_t": float(cut_t),
            "cut_a": float(cut_a),
            "eps": float(eps),
        },
    }
    prior_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    with prior_pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_idx", "y_true", "p_oof"])
        writer.writeheader()
        for eid, yy, pp in zip(ep_ids, y.tolist(), oof.tolist()):
            writer.writerow({"episode_idx": int(eid), "y_true": int(yy), "p_oof": float(pp)})

    model_obj = {
        "feature_names": FEATURE_NAMES,
        "scaler_mean": scaler_full.mean_.tolist(),
        "scaler_scale": scaler_full.scale_.tolist(),
        "coef": model_full.coef_[0].tolist(),
        "intercept": float(model_full.intercept_[0]),
        "fit_params": {
            "late_tau_start": int(late_tau_start),
            "cut_t": float(cut_t),
            "cut_a": float(cut_a),
            "eps": float(eps),
        },
        "source_run_dir": str(base_default_run_dir),
    }
    prior_model_path.write_text(json.dumps(model_obj, indent=2, ensure_ascii=True), encoding="utf-8")
    return load_prior_model(prior_model_path)


def prior_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prior_model(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    model["path"] = str(path)
    model["hash"] = prior_hash(path)
    model["feature_names"] = list(model["feature_names"])
    model["scaler_mean"] = np.asarray(model["scaler_mean"], dtype=np.float64)
    model["scaler_scale"] = np.asarray(model["scaler_scale"], dtype=np.float64)
    model["coef"] = np.asarray(model["coef"], dtype=np.float64)
    model["scaler_mean_jax"] = jnp.asarray(model["scaler_mean"], dtype=jnp.float32)
    model["scaler_scale_jax"] = jnp.asarray(model["scaler_scale"], dtype=jnp.float32)
    model["coef_jax"] = jnp.asarray(model["coef"], dtype=jnp.float32)
    model["intercept"] = float(model["intercept"])
    return model


def predict_prior_probability(model: dict[str, Any], features: np.ndarray) -> float:
    z = (features - model["scaler_mean"]) / (model["scaler_scale"] + 1e-12)
    logit = float(np.dot(model["coef"], z) + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def predict_prior_probability_jax(model: dict[str, Any], features: jnp.ndarray) -> jnp.ndarray:
    features = jnp.asarray(features, dtype=jnp.float32)
    z = (features - model["scaler_mean_jax"]) / (model["scaler_scale_jax"] + 1e-12)
    logit = jnp.dot(model["coef_jax"], z) + jnp.asarray(model["intercept"], dtype=jnp.float32)
    return jax.nn.sigmoid(logit)
