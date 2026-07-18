"""Build figure_board.tex: the curated exhibit set (user-selected)."""
import re

PAPER = "C:/Users/ofekc/Desktop/Msc/Thesis/paper"
src = open(f"{PAPER}/test.tex", encoding="utf-8").read()

def tabular_of(label):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        if label in m.group(0):
            t = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", m.group(0), re.S)
            return t.group(0)
    raise SystemExit(f"table {label} not found")

t_dense = tabular_of("tab:results_summary")
t_euler = tabular_of("tab:euler_all_engines")
t_evr = tabular_of("tab:euler_vs_radix")

def page(title, q, number, body):
    return (f"\\section*{{{title}}}\n"
            f"\\textbf{{Question:}} {q}\\\\[2pt]\n"
            f"\\textbf{{The number:}} {number}\n\\vspace{{6pt}}\n\n"
            f"{body}\n\\newpage\n")

doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{graphicx,booktabs,multirow,amsmath,amssymb}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\LARGE\bfseries Figure Board --- Selected Exhibits}\\[4pt]
{\large BvN / Radix / Euler decomposition study \quad ($n{=}256$, $k{=}256$, $W_{\max}{=}64$, dense unless noted)}\end{center}
\vspace{8pt}
"""
doc += page(r"1 --- Cross-engine trade-off: \textsc{Radix}",
    r"How do the four engines trade runtime vs.\ cycle and vs.\ permutations as the Radix base shrinks (BvN$\to$B2)?",
    r"Engines differ ${>}20\times$ in $N$ at BvN yet all converge to $N\approx657$ at $B{=}2$; runtime drops up to $60\times$ for $+38\%$ cycle.",
    r"\includegraphics[width=0.49\textwidth]{plots_pdf/tradeoff_runtime_cycle.pdf}\hfill\includegraphics[width=0.49\textwidth]{plots_pdf/tradeoff_runtime_permutations.pdf}")
doc += page(r"2 --- Cross-engine trade-off: \textsc{Euler} (end-to-end runtime, depths 1--4)",
    r"Does splitting pay end-to-end (split $+$ max-leaf), and what does it cost per engine?",
    r"GW Dynamic total $1330\to109$\,ms ($12\times$) at $C=S$ unchanged; price: $N$ $1122\to3268$; GW Static's $N$ stays flat ($\bar\lambda\approx1$ saturated).",
    r"\includegraphics[width=\textwidth]{plots_pdf/euler_tradeoff_corrected.pdf}")
doc += page(r"3 --- Winner map (ratio parameterisation, $r=T_{unit}/T_{config}$ up to 1000)",
    r"Which configuration minimises DCT at each hardware point, expressed by the ratio $r$?",
    r"Only 4 of 32 configurations ever win; on the lecturer line $r{=}10$, Euler d4 is optimal for all $T_{config}\gtrsim2\,\mu$s.",
    r"\includegraphics[width=\textwidth]{plots_pdf/dct_param_space_ratio.pdf}")
doc += page(r"4 --- Winner map as 3-D DCT surfaces",
    r"What does the completion-time landscape look like, and where do the two dominant configurations cross?",
    r"The surfaces' intersection curve \emph{is} the crossover boundary; the minimum-DCT floor is coloured by winner (legend added).",
    r"\includegraphics[width=\textwidth]{plots_pdf/dct_3d_surfaces.pdf}")
doc += page(r"5 --- Table: all engines across \textsc{Radix} (dense summary)",
    r"What does each engine + Radix base cost in runtime, cycle, permutations?",
    r"All strong engines hit $C=S=8211$ at BvN; Radix $B{=}2$ cuts Hungarian runtime $641\to69$\,ms for $+38\%$ cycle.",
    "\\begin{center}\\small\n" + t_dense + "\n\\end{center}")
doc += page(r"6 --- Table: all engines across \textsc{Euler}",
    r"How do max-leaf extraction time, cycle, and permutations evolve with split depth for every engine?",
    r"The three strong engines hold $C=S=8174$ at every depth; extraction falls ${\approx}2\times$ per level (split $<50$\,ms at depth 3).",
    "\\begin{center}\\small\n" + t_euler + "\n\\end{center}")
doc += page(r"7 --- DCT vs.\ $T_{config}$",
    r"When does the decomposition choice change completion time on real optical hardware ($T_{unit}=0.01$\,ms)?",
    r"Hungarian Euler d4 is lowest for $T_{config}\lesssim80\,\mu$s; Radix $B{=}2$ takes over beyond it.",
    r"\begin{center}\includegraphics[width=0.85\textwidth]{plots_pdf/dct_vs_tconfig.pdf}\end{center}")
doc += page(r"8 --- Table: Euler vs.\ Radix (Hungarian matching on every unit)",
    r"Head-to-head at comparable parallel-unit counts and honest accounting (Split charged): who wins on Total, at what cycle cost?",
    r"Euler d1 Total $320$\,ms vs Radix $B8$ $303$\,ms --- competitive, but Euler keeps $C=S$ ($C/S=1.0$) vs Radix's $+20\%$.",
    "\\begin{center}\\small\n" + t_evr + "\n\\end{center}")
doc += page(r"9 --- Design space $(N, C, T_{calc})$ --- total runtime",
    r"How do the two frameworks move through the full three-metric cost space that DCT prices?",
    r"Radix and Euler leave each engine's BvN anchor in \emph{orthogonal} directions: Radix trades $C\!\uparrow$ for $N\!\downarrow$; Euler trades $N\!\uparrow$ for $T_{calc}\!\downarrow$ at $C=S$. No move improves all three.",
    r"\includegraphics[width=\textwidth]{plots_pdf/design_space_projections.pdf}")
doc += "\\end{document}\n"
open(f"{PAPER}/figure_board.tex", "w", encoding="utf-8").write(doc)
print("figure_board.tex written")
