"""Build a single self-contained main.tex: inline numbers.tex, plot data and the .bbl."""
import os, re, sys
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tex")
out = sys.argv[1]
s = open(f"{T}/main.tex").read()
nums = open(f"{T}/numbers.tex").read()
s = s.replace("\\input{numbers}", nums.strip())
tables = ""
for name in ("coverage", "budget", "learning"):
    data = open(f"{T}/plotdata/{name}.dat").read().strip()
    tables += f"\\pgfplotstableread{{\n{data}\n}}\\{name}data\n"
    s = s.replace(f"{{plotdata/{name}.dat}}", f"{{\\{name}data}}")
s = s.replace("\\emergencystretch=2em", "\\emergencystretch=2em\n%% plot data (generated from results/*.json)\n" + tables.strip())
bbl = open(f"{T}/main.bbl").read()
bbl = bbl.replace("PatternsArticle 100804 (2023).", "Patterns (2023) 100804.")
s = s.replace("\\bibliographystyle{elsarticle-num}\n\\bibliography{references}",
              "%% References embedded (generated with elsarticle-num.bst from references.bib).\n" + bbl.strip())
assert "\\input{" not in s and "plotdata/" not in s and "\\bibliography{" not in s
open(out, "w").write(s)
print("wrote", out, len(s))
