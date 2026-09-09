"""IBM 実機での推論。保存済みモデル（best パラメータ）の 50 テスト点の回路を SamplerV2 で実行し、
読み出し <H> をカウントから計算してテスト誤差を出す。指標は同じ回路の厳密値（Aer 無雑音）と比較する。

使い方:
  python hardware_inference.py --backend ibm_marrakesh --models "models/d4/*seed0.json" --shots 2048 --submit
  python hardware_inference.py --collect <job_id>        # 結果回収と解析
--submit なしでは transpile 後の 2 量子ビットゲート数・深さと推定ショット総数だけを表示する。
"""
import argparse, glob, json, os, time
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import avqkan as A
from qiskit_bridge import build_circuit, readout_op, expval_from_counts, load_model

OUT = "results_hardware"; os.makedirs(OUT, exist_ok=True)


def prepare(model_paths, backend):
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=0)
    jobs = []
    for path in model_paths:
        m, cfg = load_model(path)
        prob = A.Problem(A.Config(nq=cfg.nq, seed=cfg.seed, readout_mode=cfg.readout_mode), m["target"])
        v, labels = np.array(m["best"]["vlist"]), m["best"]["ansatz"]
        circs = [pm.run(build_circuit(v, x, labels, cfg)) for x in prob.Xf]
        jobs.append(dict(model=path, target=m["target"], seed=cfg.seed, nq=cfg.nq, readout_mode=cfg.readout_mode,
                         readout=list(cfg.readout), exact_test=m["best"]["test_absdist"],
                         ff=prob.ff.tolist(), circuits=circs,
                         cz=float(np.mean([c.count_ops().get("cz", 0) for c in circs])),
                         depth=float(np.mean([c.depth() for c in circs]))))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--models", default="models/d4/*seed0.json")
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", default=None)
    a = ap.parse_args()
    svc = QiskitRuntimeService()
    if a.collect:
        meta = json.load(open(f"{OUT}/job_{a.collect}.json"))
        job = svc.job(a.collect)
        print("status:", job.status())
        res = job.result()
        k = 0
        for mj in meta["models"]:
            cfg = A.Config(nq=mj["nq"], readout_mode=mj["readout_mode"], readout=tuple(mj["readout"]))
            op = readout_op(cfg)
            h = []
            for _ in range(50):
                counts = res[k].data.c.get_counts(); k += 1
                h.append(expval_from_counts(counts, op))
            te = float(np.sum(np.abs(np.array(h) - np.array(mj["ff"]))))
            mj["hardware_test"] = te; mj["h"] = h
            print(f"{os.path.basename(mj['model'])}: exact {mj['exact_test']:.3f}  hardware {te:.3f}  (cz/circuit {mj['cz']:.1f}, depth {mj['depth']:.0f})")
        meta["usage"] = str(job.usage()) if hasattr(job, "usage") else None
        json.dump(meta, open(f"{OUT}/job_{a.collect}.json", "w"), indent=1)
        return
    backend = svc.backend(a.backend)
    jobs = prepare(sorted(glob.glob(a.models)), backend)
    ncirc = sum(len(j["circuits"]) for j in jobs)
    print(f"backend {backend.name}: {len(jobs)} models, {ncirc} circuits, {a.shots} shots each -> {ncirc * a.shots:,} shots")
    for j in jobs:
        print(f"  {os.path.basename(j['model'])}: nq={j['nq']} cz/circuit={j['cz']:.1f} depth={j['depth']:.0f}")
    if not a.submit:
        return
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = a.shots
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.twirling.enable_measure = True
    job = sampler.run([c for j in jobs for c in j["circuits"]])
    print("submitted job", job.job_id())
    meta = dict(job_id=job.job_id(), backend=backend.name, shots=a.shots, submitted=time.strftime("%Y-%m-%d %H:%M:%S"),
                models=[{k: v for k, v in j.items() if k != "circuits"} for j in jobs])
    json.dump(meta, open(f"{OUT}/job_{job.job_id()}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
