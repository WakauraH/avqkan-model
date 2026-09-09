"""分類（SI）: cls2d, seeds 0–9。誤分類数/50（訓練コスト早期停止）。古典はロジスティック回帰, SVM-RBF, 多数決。"""
import csv, os, json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from avqkan import Config, Problem

SEEDS = range(10)


def es(pat):
    out = []
    for s in SEEDS:
        p = pat.replace("SEED", str(s))
        if not os.path.exists(p):
            return None
        rows = list(csv.DictReader(open(p)))
        te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
        out.append(float(te[int(np.argmin(co))]))
    return np.array(out)


def es_tuned():
    out = []
    for s in SEEDS:
        best = None
        for L in (1, 2, 3):
            p = f"results_cls/qnn{L}_lbfgs_cls2d_seed{s}.csv"
            if not os.path.exists(p):
                return None
            rows = list(csv.DictReader(open(p)))
            te = np.array([float(r["test_absdist"]) for r in rows]); co = np.array([float(r["cost"]) for r in rows])
            if best is None or co.min() < best[1]:
                best = (float(te[int(np.argmin(co))]), co.min())
        out.append(best[0])
    return np.array(out)


arms = {"avqkan": es("results_cls/cobyla_cls2d_seedSEED.csv"),
        "avqkan_way1": es("results_cls/cobyla_cls2d_seedSEED_way1.csv"),
        "qnn_paper": es("results_cls/qnn3_cobyla_cls2d_seedSEED.csv"),
        "qnn_tuned": es_tuned()}
lr, svm, maj = [], [], []
for s in SEEDS:
    prob = Problem(Config(seed=s), "cls2d")
    X, Xf, y, yf = prob.X[:, :2], prob.Xf[:, :2], prob.f, prob.ff
    lr.append(float(np.sum(LogisticRegression().fit(X, y).predict(Xf) != yf)))
    svm.append(float(np.sum(SVC(kernel="rbf").fit(X, y).predict(Xf) != yf)))
    maj.append(float(np.sum(np.sign(np.sum(y) + 1e-9) != yf)))
arms.update(logistic=np.array(lr), svm_rbf=np.array(svm), majority=np.array(maj))
print(f"{'arm':12s} {'errors/50':>14s}  {'vs AVQKAN p':>11s}")
out = {}
for k, v in arms.items():
    if v is None:
        print(f"{k:12s} incomplete"); continue
    line = f"{k:12s} {v.mean():6.2f} ± {v.std(ddof=1):5.2f}"
    if k != "avqkan" and arms["avqkan"] is not None and not np.allclose(v, arms["avqkan"]):
        line += f"  {wilcoxon(arms['avqkan'], v)[1]:11.4f}"
    print(line); out[k] = v.tolist()
json.dump(out, open("results_model/cls.json", "w"), indent=1)
