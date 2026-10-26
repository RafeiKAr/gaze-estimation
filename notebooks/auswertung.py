"""auswertung.py – Tabellen und Vergleiche für das Notebook Masterarbeit.ipynb.

Dieses Modul TRAINIERT NICHTS. Es liest nur Ergebnisdateien und macht daraus
die Tabellen von Kapitel 4:

  pruefe_daten()             Sind Bilder und Labels aller Sessions da?
  basismodelle_tabelle()     Die zwei Basismodelle (E0): Seeds, Fehler, Baseline
  vergleiche_basismodelle()  Neu trainierte Basismodelle gegen die alten
  hyperparameter_tabellen()  Tabellen 4.3–4.9 aus exp_results_all.json
  tabelle_e1()               Tabelle 4.13: alle 18 Sessions, zwei Basismodelle
  tabelle_e1b()              Tabelle 4.14: Gen 4 mit bereinigten Labels
  tabelle_e2()               Tabelle 4.15: Random / Sequential / Two-Video
  tabelle_e3()               Tabellen 4.17 und 4.18: Session-Design
  vergleiche_mit_referenz()  Neue Läufe gegen die gespeicherten Läufe

Alle Fehler sind Diagonalfehler in % (metrics.diagonal_error_pct).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import personal_exp as pe
from metrics import diagonal_error_pct

REFERENZ = Path("results/personal/personal_runs.jsonl")        # Ergebnisse der Masterarbeit
VORHERSAGEN = Path("results/personal/predictions")
GEN = {s: g.replace("gen", "Gen ") for g, ss in pe.SESSIONS_GEN.items() for s in ss}
DESIGN = {"Gen 1": "Standard (Gen 1)", "Gen 2": "Zufällige Sprünge (Gen 2)", "Gen 4": "Ecken-gewichtet (Gen 4)"}
REGIONEN = {"Mitte": [(0.50, 0.50)],
            "Ränder": [(0.10, 0.50), (0.50, 0.10), (0.90, 0.50), (0.50, 0.90)],
            "Ecken": [(0.10, 0.10), (0.10, 0.90), (0.90, 0.10), (0.90, 0.90)]}


# ----------------------------------------------------------------------------
# kleine Hilfen
# ----------------------------------------------------------------------------
def ci98(x):
    """Mittelwert und 98-%-Konfidenzintervall (Student-t)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan, np.nan
    m = x.mean()
    h = stats.t.ppf(0.99, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def _ci_text(x):
    m, lo, hi = ci98(x)
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def laeufe_laden(datei=REFERENZ):
    """Alle Läufe einer Ergebnisdatei als Tabelle (ohne Lernkurven)."""
    runs = pe.load_runs(datei)
    if not runs:
        return pd.DataFrame()
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in runs])
    df = df[df.features == "cached"].copy()
    df["generation"] = df.session.map(GEN)
    return df


def _auswahl(df, split="random", labels="original", bases=None, seeds=(42,)):
    d = df[(df.split == split) & (df.labels == labels) & df.seed.isin(seeds)]
    return d if bases is None else d[d.base.isin(bases)]


# ----------------------------------------------------------------------------
# Daten und Basismodelle
# ----------------------------------------------------------------------------
def pruefe_daten():
    """Für jede Session: Anzahl Labels, Anzahl Bilder, Ordner, alles da?"""
    rows = []
    for s in pe.ALL_SESSIONS:
        d = pe.SESSION_ROOT / s
        img = d / "images2" if (d / "images2").is_dir() else d / "images"
        n_lab = len(pd.read_csv(d / "norm_labels.csv")) if (d / "norm_labels.csv").exists() else 0
        n_img = len(list(img.glob("*.jpg"))) if img.is_dir() else 0
        rows.append({"Session": s, "Generation": GEN[s], "Labels": n_lab, "Bilder": n_img,
                     "Ordner": img.name, "OK": n_lab > 0 and n_img >= n_lab})
    return pd.DataFrame(rows)


def basismodelle_tabelle(ordner="models/base"):
    """Übersicht der gespeicherten Basismodelle (aus den .json-Dateien)."""
    rows = []
    for split in ("norm_subject", "norm_random"):
        f = Path(ordner) / f"base_{split}.json"
        if not f.exists():
            rows.append({"Split": split, "Datei": "fehlt"})
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        best = next(r for r in m["runs"] if r["seed"] == m["selected_seed"])
        rows.append({"Split": split, "Gewählter Seed": m["selected_seed"],
                     "Validierung (%)": round(best["val_error"], 2), "Test (%)": round(best["test_error"], 2),
                     "Konstant-Baseline (%)": round(m["baseline_same_subset"]["constant"], 2),
                     "Modell vorhanden": (Path(ordner) / f"base_{split}.pt").exists(),
                     "Erstellt": m.get("created", "")})
    return pd.DataFrame(rows)


def vergleiche_basismodelle(alt_ordner="models/base", neu_ordner="models/base_neu"):
    """Jeder Seed: alter und neuer Validierungs-/Testfehler und die Differenz."""
    rows = []
    for split in ("norm_subject", "norm_random"):
        fa, fn = Path(alt_ordner) / f"base_{split}.json", Path(neu_ordner) / f"base_{split}.json"
        if not (fa.exists() and fn.exists()):
            continue
        a, n = json.loads(fa.read_text(encoding="utf-8")), json.loads(fn.read_text(encoding="utf-8"))
        na = {r["seed"]: r for r in n["runs"]}
        for r in a["runs"]:
            q = na.get(r["seed"])
            if q is None:
                continue
            rows.append({"Split": split, "Seed": r["seed"],
                         "Val alt": r["val_error"], "Val neu": q["val_error"],
                         "Test alt": r["test_error"], "Test neu": q["test_error"],
                         "Differenz Test": q["test_error"] - r["test_error"],
                         "Beste Epoche alt/neu": f"{r['best_epoch']}/{q['best_epoch']}"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# GazeCapture: Hyperparameter-Studie
# ----------------------------------------------------------------------------
GEWAEHLT = {"batch_size": "32", "learning_rate": "0.0001", "trainable_layers": "fc_only",
            "activation": "Sigmoid", "loss_function": "standard", "dataset_size": "1000_200",
            "split_type": "norm_subject"}


def hyperparameter_tabellen(datei="results/exp_reproduction/exp_results_all.json"):
    """Eine Tabelle pro Sweep: Mittelwert, 98-%-KI, Mann-Whitney-p gegen die gewählte Option."""
    res = json.loads(Path(datei).read_text(encoding="utf-8"))
    out = {}
    for sweep, wahl in GEWAEHLT.items():
        if sweep not in res:
            continue
        rows = res[sweep]
        ref = next(r for r in rows if r["config"] == wahl)
        tab = []
        for r in rows:
            p = (np.nan if r is ref else
                 stats.mannwhitneyu(r["runs"], ref["runs"], alternative="two-sided").pvalue)
            tab.append({"Option": r["config"] + (" †" if r is ref else ""),
                        "Mittel (%)": round(r["mean_test_error"], 2),
                        "98-%-KI": f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]",
                        "p vs. †": "—" if r is ref else round(p, 3),
                        "Laufzeit (min)": round(r["mean_training_time_min"], 3)})
        out[sweep] = pd.DataFrame(tab)
    return out


# ----------------------------------------------------------------------------
# Personalisierung
# ----------------------------------------------------------------------------
def tabelle_e1(df):
    """Tabelle 4.13. Gibt (Tabelle pro Session, Tabelle pro Gruppe) zurück."""
    d = _auswahl(df)
    cols = {"epoch0_test_error": "Basis", "val_error": "Val.", "test_error": "Test",
            "best_epoch": "Ep.", "sekunden": "t (s)"}
    d = d.assign(sekunden=d.minutes * 60)
    teile = []
    for base in ("norm_subject", "norm_random"):
        t = d[d.base == base].set_index("session")[list(cols)].rename(columns=cols)
        t.columns = pd.MultiIndex.from_product([[base], t.columns])
        teile.append(t)
    konst = d[d.base == "norm_subject"].set_index("session")[["const_baseline"]]
    konst.columns = pd.MultiIndex.from_tuples([("", "Konst.")])
    pro_session = pd.concat(teile + [konst], axis=1).sort_index()
    pro_session.insert(0, ("", "Gen."), pro_session.index.map(lambda s: GEN[s][-1]))

    gruppen = []
    for name, ss in [("Gen 1", pe.SESSIONS_GEN["gen1"]), ("Gen 2", pe.SESSIONS_GEN["gen2"]),
                     ("Gen 4", pe.SESSIONS_GEN["gen4"]), ("Alle (18)", pe.ALL_SESSIONS)]:
        for base in ("norm_subject", "norm_random"):
            g = d[(d.base == base) & d.session.isin(ss)]
            gruppen.append({"Gruppe": name, "Basismodell": base, "n": len(g),
                            "Basis": _ci_text(g.epoch0_test_error), "Val.": _ci_text(g.val_error),
                            "Test": _ci_text(g.test_error), "t (s)": _ci_text(g.sekunden),
                            "Konst.": _ci_text(g.const_baseline)})
    return pro_session.round(2), pd.DataFrame(gruppen)


def tests_e1(df):
    """Statistische Tests aus Abschnitt 4.4.2 als Text."""
    d = _auswahl(df)
    zeilen = []
    for base in ("norm_subject", "norm_random"):
        b = d[d.base == base].sort_values("session")
        zeilen.append(f"{base}: Test {b.test_error.mean():.2f} % | Konstant {b.const_baseline.mean():.2f} % | "
                      f"Basismodell {b.epoch0_test_error.mean():.2f} % | "
                      f"Wilcoxon Test<Konstant p={stats.wilcoxon(b.test_error, b.const_baseline).pvalue:.2g} | "
                      f"Wilcoxon Basis vs. Konstant p={stats.wilcoxon(b.epoch0_test_error, b.const_baseline).pvalue:.2f}")
    a = d[d.base == "norm_subject"].set_index("session").test_error
    c = d[d.base == "norm_random"].set_index("session").test_error
    common = a.index.intersection(c.index)
    if len(common) > 1:
        zeilen.append(f"norm_subject vs. norm_random: Wilcoxon p={stats.wilcoxon(a[common], c[common]).pvalue:.2f}")
    return "\n".join(zeilen)


def _move_wait(session, seed=42):
    """Testfehler des Original-Modells nur auf MOVE- bzw. WAIT-Frames (Gen 4)."""
    f = VORHERSAGEN / f"pred_{session}-{session}_norm_subject_original_random_standard_s{seed}.csv"
    if not f.exists():
        return np.nan, np.nan, 0
    orig = pe.load_labels(session, "original")
    move = set(pe.load_labels(session, "clean")["frame"].astype(str))
    _, _, te = pe.split_indices(len(orig), "random", seed)
    ist_move = orig.iloc[te]["frame"].astype(str).isin(move).to_numpy()
    p = pd.read_csv(f)
    gt, pr = p[["gt_x", "gt_y"]].to_numpy(), p[["pred_x", "pred_y"]].to_numpy()
    return diagonal_error_pct(gt[ist_move], pr[ist_move]), diagonal_error_pct(gt[~ist_move], pr[~ist_move]), int((~ist_move).sum())


def tabelle_e1b(df):
    """Tabelle 4.14: Original (alle / MOVE / WAIT) gegen bereinigte Labels."""
    o = _auswahl(df, bases=["norm_subject"]).set_index("session")
    c = _auswahl(df, labels="clean", bases=["norm_subject"]).set_index("session")
    rows = []
    for s in pe.SESSIONS_GEN["gen4"]:
        if s not in o.index or s not in c.index:
            continue
        move, wait, n_w = _move_wait(s)
        rows.append({"Session": s, "Alle": o.at[s, "test_error"], "MOVE": move, "WAIT": wait, "n_W": n_w,
                     "Bereinigt": c.at[s, "test_error"], "Konst.": c.at[s, "const_baseline"],
                     "Δ": c.at[s, "test_error"] - move})
    t = pd.DataFrame(rows).set_index("Session")
    if len(t):
        t.loc["Mittel"] = t.mean()
    return t.round(2)


def tabelle_e2(df):
    """Tabelle 4.15: drei Strategien zum Aufteilen der Daten (Sessions 21/22, 3 Seeds)."""
    d = df[df.session.isin(["21", "22"]) & (df.base == "norm_subject") & (df.labels == "original")
           & df.seed.isin([42, 43, 44])]
    t = (d.groupby(["split", "session", "test_session"])
          .agg(Test=("test_error", "mean"), SD=("test_error", "std"), Val=("val_error", "mean"),
               Basis=("epoch0_test_error", "mean"), Konst=("const_baseline", "mean"),
               n_train=("n_train", "first"), n_val=("n_val", "first"), n_test=("n_test", "first"),
               t_s=("minutes", lambda m: 60 * m.mean()), Laeufe=("seed", "count")))
    t["Red. (%)"] = 100 * (1 - t.Test / t.Konst)
    return t.round(2)


def _region_fehler(gt, pr):
    out = {}
    for name, punkte in REGIONEN.items():
        m = np.zeros(len(gt), bool)
        for q in punkte:
            m |= np.linalg.norm(gt - np.array(q), axis=1) <= 0.15
        out[name] = (diagonal_error_pct(gt[m], pr[m]) if m.any() else np.nan, int(m.sum()))
    return out


def tabelle_e3(df):
    """Tabellen 4.17 (Fehler pro Gestaltung) und 4.18 (Fehler pro Bildschirmregion)."""
    r = _auswahl(df, bases=["norm_subject"]).set_index("session")
    q = _auswahl(df, split="sequential", bases=["norm_subject"]).set_index("session")
    pro_design = []
    for gen in ("Gen 1", "Gen 2", "Gen 4"):
        ss = [s for s in pe.ALL_SESSIONS if GEN[s] == gen and s in q.index]
        pro_design.append({"Gestaltung": DESIGN[gen], "Test zufällig": r.loc[ss, "test_error"].mean(),
                           "Konst. zufällig": r.loc[ss, "const_baseline"].mean(),
                           "Test sequenziell": q.loc[ss, "test_error"].mean(),
                           "Konst. sequenziell": q.loc[ss, "const_baseline"].mean(),
                           "Ep. 0": f"{int((q.loc[ss, 'best_epoch'] == 0).sum())} / {len(ss)}",
                           "n Test": q.loc[ss, "n_test"].mean()})
    ss = [s for s in pe.ALL_SESSIONS if s in q.index]
    pro_design.append({"Gestaltung": "Alle (18)", "Test zufällig": r.loc[ss, "test_error"].mean(),
                       "Konst. zufällig": r.loc[ss, "const_baseline"].mean(),
                       "Test sequenziell": q.loc[ss, "test_error"].mean(),
                       "Konst. sequenziell": q.loc[ss, "const_baseline"].mean(),
                       "Ep. 0": f"{int((q.loc[ss, 'best_epoch'] == 0).sum())} / {len(ss)}",
                       "n Test": q.loc[ss, "n_test"].mean()})

    regionen = []
    for split in ("random", "sequential"):
        werte = []
        for s in ss:
            f = VORHERSAGEN / f"pred_{s}-{s}_norm_subject_original_{split}_standard_s42.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            gt, pr = p[["gt_x", "gt_y"]].to_numpy(), p[["pred_x", "pred_y"]].to_numpy()
            e = {"session": s, "gen": GEN[s], "Alle": diagonal_error_pct(gt, pr)}
            for name, (fehler, n) in _region_fehler(gt, pr).items():
                e[name], e[f"n {name}"] = fehler, n
            werte.append(e)
        w = pd.DataFrame(werte)
        if w.empty:
            continue
        for gen, g in list(w.groupby("gen")) + [("Alle (18)", w)]:
            zeile = {"Split": split, "Gestaltung": DESIGN.get(gen, gen), "Alle": g["Alle"].mean()}
            for name in REGIONEN:
                zeile[name] = f"{g[name].mean():.1f} ({int(g[f'n {name}'].sum())})"
            regionen.append(zeile)
    return pd.DataFrame(pro_design).round(2), pd.DataFrame(regionen).round(1)


# ----------------------------------------------------------------------------
# Vergleich neuer Läufe mit den gespeicherten Ergebnissen
# ----------------------------------------------------------------------------
def vergleiche_mit_referenz(neu_datei, ref_datei=REFERENZ):
    """Jeder neue Lauf neben dem gleichen gespeicherten Lauf (gleiche Session,
    Basismodell, Labels, Split, Seed). Gibt (Tabelle, größte Abweichung) zurück."""
    neu, ref = laeufe_laden(neu_datei), laeufe_laden(ref_datei)
    if neu.empty:
        return pd.DataFrame(), np.nan
    key = ["session", "test_session", "base", "labels", "split", "seed"]
    m = neu.merge(ref, on=key, suffixes=("_neu", "_alt"))
    t = pd.DataFrame({
        "Session": m.session, "Test-Session": m.test_session, "Basismodell": m.base,
        "Labels": m.labels, "Split": m.split, "Seed": m.seed,
        "Test alt": m.test_error_alt, "Test neu": m.test_error_neu,
        "Differenz": m.test_error_neu - m.test_error_alt,
        "Epoche alt/neu": m.best_epoch_alt.astype(str) + "/" + m.best_epoch_neu.astype(str)})
    return t.round(4), float(t.Differenz.abs().max()) if len(t) else np.nan


def abbildung_sessions(df, datei="results/figures/fig_sessions_baseline.png"):
    """Abbildung 4.4: Testfehler pro Session (Konstant-Baseline und beide Basismodelle)."""
    import matplotlib.pyplot as plt
    d = _auswahl(df)
    S = sorted(d.session.unique())
    x, w = np.arange(len(S)), 0.27
    v = lambda base, col: d[d.base == base].set_index("session").reindex(S)[col]
    fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
    ax.bar(x - w, v("norm_subject", "const_baseline"), w, color="#9e9e9e", label="Constant baseline")
    ax.bar(x, v("norm_subject", "test_error"), w, color="#1f77b4", label="Personalized (norm_subject)")
    ax.bar(x + w, v("norm_random", "test_error"), w, color="#ff7f0e", label="Personalized (norm_random)")
    ax.set_xticks(x, S)
    ax.set_xlabel("Session")
    ax.set_ylabel("Diagonal error (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.legend(loc="outside lower center", ncol=3, fontsize=8, frameon=False)
    Path(datei).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(datei, dpi=200)
    return fig
