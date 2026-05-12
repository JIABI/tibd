import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12


def rankdata_average(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    rx = rankdata_average(np.asarray(x))
    ry = rankdata_average(np.asarray(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum()) + EPS
    return float((rx * ry).sum() / denom)


def partial_spearman(x, y, controls):
    xr = rankdata_average(np.asarray(x))
    yr = rankdata_average(np.asarray(y))
    Z = np.column_stack([rankdata_average(np.asarray(c)) for c in controls])
    Z = np.column_stack([np.ones(len(xr)), Z])
    bx, *_ = np.linalg.lstsq(Z, xr, rcond=None)
    by, *_ = np.linalg.lstsq(Z, yr, rcond=None)
    rx = xr - Z @ bx
    ry = yr - Z @ by
    denom = np.sqrt((rx**2).sum() * (ry**2).sum()) + EPS
    return float((rx * ry).sum() / denom)


def sample_smooth(p, K=20, alpha=0.5, rng=None):
    if K is None:
        return p.copy()
    rng = np.random.default_rng() if rng is None else rng
    counts = rng.multinomial(K, p)
    post = counts + alpha
    return post / post.sum()


def tibd_decompose(probs):
    s = np.sqrt(np.clip(probs, EPS, 1.0))
    mu = s.mean(axis=(0, 1))
    a = s.mean(axis=1) - mu[None, :]
    b = s.mean(axis=0) - mu[None, :]
    c = s - mu[None, None, :] - a[:, None, :] - b[None, :, :]

    U_ling = float((a**2).sum(axis=1).mean())
    U_epi = float((b**2).sum(axis=1).mean())
    U_x = float((c**2).sum(axis=-1).mean())
    U_total = U_ling + U_epi + U_x
    U_total_empirical = float(((s - mu[None, None, :]) ** 2).sum(axis=-1).mean())
    decomp_error = abs(U_total_empirical - U_total)
    if decomp_error > 1e-8:
        print(f"[warn] decomposition error: {decomp_error:.3e}")

    p_bar = probs.mean(axis=(0, 1))
    U_sem = float(1.0 - np.sum(np.sqrt(np.clip(p_bar, EPS, 1.0))) / np.sqrt(len(p_bar)))
    return {
        "U_ling": U_ling,
        "U_epi": U_epi,
        "U_x": U_x,
        "U_total": U_total,
        "U_total_empirical": U_total_empirical,
        "decomp_error": decomp_error,
        "U_sem": U_sem,
    }


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def toy_A(gamma=0.0, beta_l=None, beta_e=None, noise_matrix=None, N=4, M=4, C=2, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    beta_l = rng.normal(0.0, 1.0, N) if beta_l is None else beta_l
    beta_e = rng.normal(0.0, 1.0, M) if beta_e is None else beta_e
    noise_matrix = rng.normal(0.0, 0.1, (N, M)) if noise_matrix is None else noise_matrix
    probs = np.zeros((N, M, C), dtype=float)
    for i in range(N):
        for j in range(M):
            logit = beta_l[i] + beta_e[j] + gamma * beta_l[i] * beta_e[j] + noise_matrix[i, j]
            p1 = 1.0 / (1.0 + np.exp(-logit))
            probs[i, j] = np.array([1 - p1, p1])
    return probs


def toy_B(rho=0.0, v=None, gate_noise=None, p0=None, N=4, M=4, C=3, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    p0 = softmax(rng.normal(0, 0.4, C)) if p0 is None else p0
    v = rng.normal(0, 1.0, (M, C)) if v is None else v
    gate_noise = rng.normal(0, 0.2, N) if gate_noise is None else gate_noise
    g = 1.0 / (1.0 + np.exp(-(rho * np.linspace(-1, 1, N) + gate_noise)))
    probs = np.zeros((N, M, C), dtype=float)
    for i in range(N):
        for j in range(M):
            z = np.log(np.clip(p0, EPS, 1.0)) + g[i] * v[j]
            probs[i, j] = softmax(z)
    return probs, float(np.var(g))


def toy_ambiguous(N=4, M=4, C=2, rng=None, jitter=0.02):
    rng = np.random.default_rng() if rng is None else rng
    base = np.ones(C) / C
    eps = rng.normal(0, jitter, size=C)
    eps = eps - eps.mean()
    p0 = np.clip(base + eps, 1e-6, 1.0)
    p0 = p0 / p0.sum()
    return np.tile(p0, (N, M, 1))


def interaction_permutation_null(probs, B=100, axis="evidence", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    obs = tibd_decompose(probs)["U_x"]
    vals = []
    for _ in range(B):
        p = probs.copy()
        if axis == "evidence":
            for i in range(p.shape[0]):
                p[i] = p[i, rng.permutation(p.shape[1]), :]
        else:
            for j in range(p.shape[1]):
                p[:, j, :] = p[rng.permutation(p.shape[0]), j, :]
        vals.append(tibd_decompose(p)["U_x"])
    vals = np.asarray(vals)
    m, s = float(vals.mean()), float(vals.std())
    return {"U_x_perm_mean": m, "U_x_perm_std": s, "U_x_excess": obs - m, "U_x_z": (obs - m) / (s + EPS)}


def experiment_A_paired(gammas, Q=400, K=None, alpha=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        bl = rng.normal(0, 1, 4)
        be = rng.normal(0, 1, 4)
        noise = rng.normal(0, 0.1, (4, 4))
        for g in gammas:
            p = toy_A(gamma=g, beta_l=bl, beta_e=be, noise_matrix=noise, rng=rng)
            if K is not None:
                p = np.array([[sample_smooth(p[i, j], K=K, alpha=alpha, rng=rng) for j in range(4)] for i in range(4)])
            d = tibd_decompose(p)
            d.update({"toy": "A", "q": q, "coupling": g, "abs_coupling": abs(g), "K": K or 0, "alpha": alpha})
            rows.append(d)
    df = pd.DataFrame(rows)
    rho = spearman_corr(df["U_x"], df["abs_coupling"])
    prho = partial_spearman(df["U_x"], df["abs_coupling"], [df["U_ling"], df["U_epi"], df["U_sem"]])
    mono = []
    for _, gdf in df.sort_values("coupling").groupby("q"):
        u = gdf["U_x"].to_numpy()
        mono.append(bool(np.all(np.diff(u) >= -1e-8)))
    return df, {"spearman": rho, "partial_spearman": prho, "monotonicity_rate": float(np.mean(mono))}


def experiment_B_paired(rhos, Q=400, K=None, alpha=0.5, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(Q):
        p0 = softmax(rng.normal(0, 0.5, 3))
        v = rng.normal(0, 1.0, (4, 3))
        gn = rng.normal(0, 0.25, 4)
        for rho in rhos:
            p, varg = toy_B(rho=rho, p0=p0, v=v, gate_noise=gn, rng=rng)
            if K is not None:
                p = np.array([[sample_smooth(p[i, j], K=K, alpha=alpha, rng=rng) for j in range(4)] for i in range(4)])
            d = tibd_decompose(p)
            d.update({"toy": "B", "q": q, "coupling": rho, "Var_g": varg, "K": K or 0, "alpha": alpha})
            rows.append(d)
    df = pd.DataFrame(rows)
    rho = spearman_corr(df["U_x"], df["Var_g"])
    prho = partial_spearman(df["U_x"], df["Var_g"], [df["U_ling"], df["U_epi"], df["U_sem"]])
    mono = []
    for _, gdf in df.sort_values("coupling").groupby("q"):
        mono.append(bool(np.all(np.diff(gdf["U_x"].to_numpy()) >= -1e-8)))
    return df, {"spearman": rho, "partial_spearman": prho, "monotonicity_rate": float(np.mean(mono))}


def make_routing_dataset(n_per=250, seed=7, perm_null=False, perm_B=100):
    rng = np.random.default_rng(seed)
    rows = []
    labels = ["prompt_fragile", "knowledge_gap", "coupled", "ambiguous"]
    for y in labels:
        for _ in range(n_per):
            if y == "prompt_fragile":
                p = toy_A(gamma=0.0, beta_l=rng.normal(0, 1.5, 4), beta_e=rng.normal(0, 0.1, 4), noise_matrix=rng.normal(0,0.05,(4,4)), rng=rng)
            elif y == "knowledge_gap":
                p = toy_A(gamma=0.0, beta_l=rng.normal(0, 0.1, 4), beta_e=rng.normal(0, 1.5, 4), noise_matrix=rng.normal(0,0.05,(4,4)), rng=rng)
            elif y == "coupled":
                p = toy_A(gamma=1.2, beta_l=rng.normal(0, 1.0, 4), beta_e=rng.normal(0, 1.0, 4), noise_matrix=rng.normal(0,0.02,(4,4)), rng=rng)
            else:
                p = toy_ambiguous(C=3, rng=rng)
            d = tibd_decompose(p)
            if perm_null:
                d.update(interaction_permutation_null(p, B=perm_B, rng=rng))
            d["label"] = y
            rows.append(d)
    return pd.DataFrame(rows)


def split_train_test(df, seed=0, frac=0.7):
    rng = np.random.default_rng(seed)
    train_idx = []
    for _, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        k = int(len(idx) * frac)
        train_idx.extend(idx[:k])
    train = df.loc[sorted(train_idx)].copy()
    test = df.drop(train.index).copy()
    return train, test


def evaluate_routing(df, seed=0):
    feats = ["U_ling", "U_epi", "U_x", "U_sem"]
    train, test = split_train_test(df, seed=seed)
    methods = {}

    # A argmax z-score
    mu, sd = train[feats].mean(), train[feats].std().replace(0, 1)
    zt = (train[feats] - mu) / sd
    zq = (test[feats] - mu) / sd
    fmap = {0: "prompt_fragile", 1: "knowledge_gap", 2: "coupled", 3: "ambiguous"}
    p_tr = zt.to_numpy().argmax(axis=1)
    p_te = zq.to_numpy().argmax(axis=1)
    methods["decomposition_argmax"] = (train["label"].to_numpy(), np.vectorize(fmap.get)(p_tr), test["label"].to_numpy(), np.vectorize(fmap.get)(p_te))

    # B nearest centroid
    cents = train.groupby("label")[feats].mean()
    def nearest(X):
        y=[]
        for _,r in X.iterrows():
            d=((cents-r.values)**2).sum(axis=1)
            y.append(d.idxmin())
        return np.array(y)
    methods["decomposition_centroid"] = (train["label"].to_numpy(), nearest(train[feats]), test["label"].to_numpy(), nearest(test[feats]))

    # scalar quantile maps
    def quantile_map(trv, trl, tev, q=4):
        bins = np.quantile(trv, np.linspace(0,1,q+1)); bins[0]-=1e-9; bins[-1]+=1e-9
        lut = {}
        for i in range(q):
            m=(trv>=bins[i])&(trv<bins[i+1])
            lut[i]=pd.Series(trl[m]).mode().iloc[0] if m.sum() else pd.Series(trl).mode().iloc[0]
        tri=np.clip(np.digitize(trv,bins)-1,0,q-1); tei=np.clip(np.digitize(tev,bins)-1,0,q-1)
        return np.array([lut[i] for i in tri]), np.array([lut[i] for i in tei])

    trp, tep = quantile_map(train["U_sem"].to_numpy(), train["label"].to_numpy(), test["U_sem"].to_numpy())
    methods["same-grid scalar baseline (U_sem quartile)"] = (train["label"].to_numpy(), trp, test["label"].to_numpy(), tep)
    trp, tep = quantile_map(train["U_total"].to_numpy(), train["label"].to_numpy(), test["U_total"].to_numpy())
    methods["collapsed-grid scalar route (U_total quartile)"] = (train["label"].to_numpy(), trp, test["label"].to_numpy(), tep)

    best = None
    best_acc = -1
    for s in ["U_sem", "U_total", "U_ling", "U_epi", "U_x"]:
        trp, tep = quantile_map(train[s].to_numpy(), train["label"].to_numpy(), test[s].to_numpy())
        acc = (trp == train["label"].to_numpy()).mean()
        if acc > best_acc:
            best_acc = acc
            best = (s, trp, tep)
    methods[f"best single-component scalar ({best[0]})"] = (train["label"].to_numpy(), best[1], test["label"].to_numpy(), best[2])

    records, preds = [], []
    classes = sorted(df["label"].unique())
    for name, (ytr, ptr, yte, pte) in methods.items():
        cm = pd.crosstab(pd.Series(yte, name="true"), pd.Series(pte, name="pred"), dropna=False).reindex(index=classes, columns=classes, fill_value=0)
        per_cls = {c: float(cm.loc[c, c] / max(1, cm.loc[c].sum())) for c in classes}
        records.append({"method": name, "train_acc": float((ytr == ptr).mean()), "test_acc": float((yte == pte).mean()), "confusion_matrix": cm.to_dict(), "per_class_acc": per_cls})
        preds.extend([{"method":name, "split":"test", "true":t, "pred":p} for t,p in zip(yte,pte)])
    return pd.DataFrame(records), pd.DataFrame(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Q", type=int, default=400)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--perm-null", action="store_true")
    ap.add_argument("--perm-B", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gammas = np.linspace(0, 1.5, 7)
    rhos = np.linspace(0, 2.0, 7)

    dfA_p, mA_p = experiment_A_paired(gammas, Q=args.Q, K=None, alpha=args.alpha, seed=args.seed)
    dfA_s, mA_s = experiment_A_paired(gammas, Q=args.Q, K=args.K, alpha=args.alpha, seed=args.seed)
    dfB_p, mB_p = experiment_B_paired(rhos, Q=args.Q, K=None, alpha=args.alpha, seed=args.seed + 1)
    dfB_s, mB_s = experiment_B_paired(rhos, Q=args.Q, K=args.K, alpha=args.alpha, seed=args.seed + 1)

    alpha_vals = [0.1, 0.5, 1.0]
    alpha_rows = []
    for a in alpha_vals:
        _, ma = experiment_A_paired(gammas, Q=max(100, args.Q // 2), K=args.K, alpha=a, seed=args.seed)
        dfa, _ = experiment_A_paired(gammas, Q=max(100, args.Q // 2), K=args.K, alpha=a, seed=args.seed)
        _, mb = experiment_B_paired(rhos, Q=max(100, args.Q // 2), K=args.K, alpha=a, seed=args.seed + 1)
        dfb, _ = experiment_B_paired(rhos, Q=max(100, args.Q // 2), K=args.K, alpha=a, seed=args.seed + 1)
        alpha_rows += [
            {"toy": "A", "alpha": a, **ma, "mean_Ux": float(dfa["U_x"].mean())},
            {"toy": "B", "alpha": a, **mb, "mean_Ux": float(dfb["U_x"].mean())},
        ]
    df_alpha = pd.DataFrame(alpha_rows)

    routing_df = make_routing_dataset(seed=args.seed + 2, perm_null=args.perm_null, perm_B=args.perm_B)
    routes, route_preds = evaluate_routing(routing_df, seed=args.seed)

    raw = pd.concat([
        dfA_p.assign(source="toyA_population_paired"),
        dfA_s.assign(source="toyA_sampled_paired"),
        dfB_p.assign(source="toyB_population_paired"),
        dfB_s.assign(source="toyB_sampled_paired"),
        routing_df.assign(source="routing_features"),
    ], ignore_index=True, sort=False)
    raw.to_csv(out / "tibd_synthetic_raw.csv", index=False)
    route_preds.to_csv(out / "tibd_synthetic_routing_predictions.csv", index=False)

    summary = {
        "toy_A_population": mA_p,
        "toy_A_sampled": mA_s,
        "toy_B_population": mB_p,
        "toy_B_sampled": mB_s,
        "alpha_sensitivity": df_alpha.to_dict(orient="records"),
        "routing": routes.to_dict(orient="records"),
        "decomposition_max_error": float(raw["decomp_error"].dropna().max()),
        "labels": ["same-grid scalar baseline", "decomposition routing"],
    }
    (out / "tibd_synthetic_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_plots:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for comp in ["U_ling", "U_epi", "U_x"]:
            axes[0, 0].plot(dfA_p.groupby("coupling")[comp].mean(), label=comp)
        axes[0, 0].set_title("Toy A paired components vs gamma")
        axes[0, 0].legend()
        axes[0, 1].scatter(dfA_p["abs_coupling"], dfA_p["U_x"], s=5, alpha=0.3)
        axes[0, 1].set_title("Toy A U_x vs |gamma|")
        for comp in ["U_ling", "U_epi", "U_x"]:
            axes[0, 2].plot(dfB_p.groupby("coupling")[comp].mean(), label=comp)
        axes[0, 2].set_title("Toy B paired components vs rho")
        axes[0, 2].legend()
        axes[1, 0].scatter(dfB_p["Var_g"], dfB_p["U_x"], s=5, alpha=0.3)
        axes[1, 0].set_title("Toy B U_x vs Var(g)")
        axes[1, 1].bar(routes["method"], routes["test_acc"])
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].set_title("Routing test accuracy")
        best_cm = routes.sort_values("test_acc", ascending=False).iloc[0]["confusion_matrix"]
        cm = pd.DataFrame(best_cm)
        axes[1, 2].imshow(cm.values, cmap="Blues")
        axes[1, 2].set_xticks(range(len(cm.columns))); axes[1, 2].set_xticklabels(cm.columns, rotation=45)
        axes[1, 2].set_yticks(range(len(cm.index))); axes[1, 2].set_yticklabels(cm.index)
        axes[1, 2].set_title("Routing confusion matrix (test)")
        plt.tight_layout()
        plt.savefig(out / "tibd_synthetic_results.png", dpi=180)

    print("=== TIBD Synthetic Summary ===")
    print(f"Toy A sampled Spearman(U_x, |gamma|): {mA_s['spearman']:.3f}")
    print(f"Toy B sampled Spearman(U_x, Var(g)): {mB_s['spearman']:.3f}")
    print(routes[["method", "train_acc", "test_acc"]].sort_values("test_acc", ascending=False).to_string(index=False))
    print(f"Saved outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
