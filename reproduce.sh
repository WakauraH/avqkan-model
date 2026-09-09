#!/usr/bin/env bash
# Re-run every analysis and regenerate every figure and table of the paper from the shipped result files.
# Takes a few minutes on a laptop. No quantum hardware access is needed.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
echo "== tests";                 python3 test_avqkan.py
echo "== qiskit bridge";         python3 qiskit_bridge.py --test
echo "== d=4 table + ablation";  python3 analyze_model.py > results_model/analyze_model.log
echo "== pre-registered study";  python3 analyze_model_confirm.py | tee results_model/confirm_analysis.txt
echo "== regularised classical (d=12, seeds 30-49)"
python3 classical_regularized.py sepn12 12 10 30-49 results_model_confirm/cobyla_sepn12_seedSEED_nq12_m30.csv
echo "== sample-size sweep";     python3 analyze_nsweep.py
echo "== classification";        python3 analyze_cls.py
echo "== noise + hardware";      python3 analyze_noise.py
echo "== figures";               python3 gen_model_figs.py
echo "done: figs_model/, results_model/"
