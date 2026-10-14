# scrap-fatigue-calibration

Code and results for the manuscript *When Steel Comes from Scrap: Batch Shift, Copper Extrapolation, and Calibrated Life Bands in Data-Driven Fatigue Prediction*.

The study asks whether data-driven fatigue-strength models trained on clean steel heats stay accurate, and keep honest uncertainty, on heats with a scrap-like chemistry (residual Cu, P, S and inclusions). It uses the public NIMS rotating-bending fatigue data (437 specimens, 155 heats).

## Data

The data are third-party and are not redistributed here. `./get_data.sh` fetches the file released with Agrawal et al. (2014), *Integrating Materials and Manufacturing Innovation* 3:8, doi:10.1186/2193-9772-3-8, from a public mirror and places it in `data/`. You can also set `NIMS_CSV=/path/to/40192_2013_16_MOESM1_ESM.csv`.

## Reproduce

```
pip install -r requirements.txt
./get_data.sh
python e1_accuracy.py 20    # point accuracy under row / heat / tramp-shift splits   (~10 min)
python e2_conformal.py 50   # calibrated intervals, P1-P4                            (~4 min)
python e3_budget.py 40      # batch budget and learning curve                        (~2 min)
python e5_cu.py 200         # copper shift vs null, class control, Cu intervals      (~5 min)
python e6_onesided.py       # one-sided bounds and side of misses                    (~3 min)
python make_numbers.py      # writes tex/numbers.tex and tex/plotdata/*.dat
```

Then compile `tex/main.tex` with `pdflatex` and `bibtex`, or run `python build_single.py main_single.tex` to produce one self-contained file with the numbers, plot data and references embedded.

| Script | Paper section |
|---|---|
| `e1_accuracy.py` | Table 2 |
| `e5_cu.py` | Table 3, Cu rows of Table 4 |
| `e2_conformal.py` | Table 4, Fig. 2, weighted-conformal gate |
| `e6_onesided.py` | Section 5.5 |
| `e3_budget.py` | Table 5, Figs. 3 and 4 |

`results/` holds the JSON outputs used in the manuscript. All random draws are seeded; tree models are single-threaded for reproducibility.

## Definitions

- **Heat**: specimens with identical composition and inclusion descriptors (155 heats).
- **Scrap signature** `v(x)`: mean percentile rank of residual Cu, P, S and total inclusion area fraction.
- **Target**: natural log of fatigue strength at 10^7 cycles.

## License

MIT (code). The NIMS data remain subject to their original terms.
