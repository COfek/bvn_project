"""Build meeting_doc.tex: the high-level package for the advisor meeting.
Part A: paper structure + story. Part B: the 9 exhibits. Part C: theoretical directions."""
import re

PAPER = "C:/Users/ofekc/Desktop/Msc/Thesis/paper"
src = open(f"{PAPER}/test.tex", encoding="utf-8").read()

def tabular_of(label):
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        if label in m.group(0):
            return re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", m.group(0), re.S).group(0)
    raise SystemExit(f"table {label} not found")

t_dense = tabular_of("tab:results_summary")
t_euler = tabular_of("tab:euler_all_engines")
t_evr = tabular_of("tab:euler_vs_radix")

def exhibit(title, q, number, body):
    return (f"\\subsection*{{{title}}}\n"
            f"\\textbf{{Question:}} {q}\\\\[2pt]\n"
            f"\\textbf{{The number:}} {number}\n\\vspace{{4pt}}\n\n"
            f"{body}\n\\newpage\n")

doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{graphicx,booktabs,multirow,amsmath,amssymb,enumitem}
\setlength{\parindent}{0pt}
\begin{document}

\begin{center}
{\LARGE\bfseries Thesis Meeting Package}\\[4pt]
{\large Permutation-Matrix Decomposition of Doubly-Stochastic Traffic Matrices\\
for Reconfigurable Switch Scheduling}\\[6pt]
{\normalsize Ofek Cohen \quad---\quad benchmark setting: $n{=}256$, $k{=}256$, $W_{\max}{=}64$, dense unless noted}
\end{center}
\vspace{6pt}
\hrule
\vspace{10pt}

\section*{Part A --- Paper structure and the story}

\textbf{The story, step by step:}
\begin{enumerate}[itemsep=2pt]
  \item Scheduling a reconfigurable switch is the problem of decomposing a demand matrix into
        weighted permutation matrices.
  \item When the demand matrix is modelled as a doubly stochastic matrix, the Birkhoff--von
        Neumann (BvN) algorithm expresses it as a sum of weighted permutation matrices whose
        weights sum to exactly $1$ (or to $S$ for a scaled doubly stochastic matrix). The sum of
        the weights is the \textbf{cycle length}; the number of permutations is the
        \textbf{permutation count}. Finding the decomposition with the fewest permutations is
        NP-hard.
  \item BvN is iterative: extract a permutation from the matrix, take the smallest entry among
        the cells selected by that permutation as its weight ($\lambda = \min$ over the selected
        entries), subtract the weighted permutation, and repeat until the matrix is zeroed out.
  \item The problem can be modelled as extracting perfect matchings from a bipartite graph:
        rows and columns are the two vertex sets, with an edge between row $i$ and column $j$
        whenever $A_{ij} > 0$.
  \item Extracting a perfect matching of that graph is extracting a permutation of the demand
        matrix --- so within this iterative extraction framework we have a free hand in choosing
        the matching algorithm.
  \item Related work --- ``Chronos: Prescheduled circuit switching for LLM training'' ---
        proposed the Wavefront Arbiter (WFA) as the matching algorithm, because it can be
        implemented in hardware and therefore runs much faster. The price, as reported in
        Chronos: the cycle length rises by a factor of ${\sim}1.2$ over $S$ --- the schedule
        takes $1.2\times$ longer to transmit, in the hope that the saved computation time more
        than covers it.
  \item Chronos further suggested parallelising the computation by rewriting the matrix:
        express every element by its base-2 bit representation, splitting the matrix into
        $\log_2(k)$ bit-planes ($k$ = the largest entry), decompose each plane in parallel, and
        merge all resulting permutations at the end.
  \item Summing up the prior art: bit-planes for parallelism, WFA for fast extraction. Two
        problems: (a) after splitting into bit-planes, nothing guarantees the planes remain
        doubly stochastic --- degrading cycle length further; (b) WFA is a \emph{maximal}
        matching engine, so it can produce partial matchings by itself (an early cell selection
        can block later cells that a full permutation would have needed).
  \item \textbf{What we propose:} a set of different matching algorithms and a set of different
        parallelisation frameworks, demonstrating a three-way trade-off between permutation
        count, cycle length, and the end-to-end runtime of the decomposition itself.
  \item We formalise the objective as the \textbf{Demand Completion Time}:
        $\mathrm{DCT} = T_{calc} + N\cdot T_{config} + C\cdot T_{unit}$ --- computation time,
        plus configurations times reconfiguration time, plus cycle length times the demand
        unit's time.
  \item \textbf{Our first extension:} instead of representing the matrix with bit-planes only,
        we rename and generalise the method as \textbf{Radix decomposition}: represent each
        entry in a chosen base --- $2, 4, 8, 16$, and so on --- splitting the matrix into digit
        planes of that base, and map the resulting Pareto over our metrics. As with base~2, the
        planes are not guaranteed to be doubly stochastic, so cycle-length optimality is never
        guaranteed, unlike in the classical algorithm.
  \item We then drew that Pareto with several matching engines: the classical Hungarian
        algorithm; the suggested WFA; a greedy scan with weighted matching that maximises the
        current step size (the weight subtracted at each iteration); and a greedy scan sorted
        once at the start --- imposing structure in the hope of beating WFA.
  \item \textbf{Our second framework} uses the Euler-tour principle. The motive: split the
        matrix into sub-matrices (leaves) that \emph{remain} doubly stochastic --- preserving
        cycle-length optimality --- with a split that is itself cheap. Following Cole \&
        Hopcroft (1982), an Euler tour on a graph where every vertex has an even number of
        edges, coloured in two alternating colours along the way, yields two matrices with
        exactly half the row/column sums ($S/2$ in the scaled case).
  \item We wanted something more sophisticated than the plain halving that alternating colours
        trivially give, so we defined a heuristic applied during the tour: when the tour goes
        from row-node $i$ to column-node $j$, colour \emph{all} the edges between those two
        nodes with the same colour. This yields sparser leaves, and we show that sparser
        matrices decompose faster end-to-end (fewer entries to zero out).
  \item Finally we present results and experiments exposing the trade-offs we found. A network
        engineer can take the DCT formula, locate themselves in the design space, and choose
        the best framework $+$ matching algorithm $+$ step strategy, getting the most out of
        their datacenter topology and scheduling.
\end{enumerate}

\medskip
\textbf{Chapters} (updated per last meeting):
\begin{enumerate}[itemsep=1pt]
  \item \textbf{Introduction} --- the reconfiguration tax; contributions.
  \item \textbf{Background \& Related Work} --- circuit-switch scheduling, BvN theorem, matching
        theory; Chronos (Radix $B{=}2$ + WFA) as the starting point of the story.
  \item \textbf{Problem Definition \& Model} --- the matrices we work on ($K$-regular scaled
        doubly stochastic, integer entries $\le W_{\max}$); the decomposition space we live in;
        the DCT objective; \emph{subsection: the three-way trade-off} ($N$, $C$, $T_{calc}$).
        Unifying remark: plain BvN is the degenerate case of \emph{both} frameworks --- Euler at
        depth $0$ and Radix with base $>$ the maximum entry.
  \item \textbf{Solution Framework} --- one chapter for the full toolbox: the matching engines
        (Hungarian, WFA, GW Dynamic, GW Static; augmenting paths), the decomposition frameworks
        (Radix digit-planes; Euler splitting with the same-direction heuristic and $O(1)$-selection
        implementation), and the step-size strategies.
  \item \textbf{Experimental Evaluation} --- the three-way trade-off, dense \& sparse, all
        engines; includes the step-strategy trade-off.
  \item \textbf{DCT Analysis} --- winner maps at realistic hardware values; sensitivity to
        computation time ($T_{calc}$, $T_{calc}/2$, $T_{calc}{=}0$ prescheduled); time to first
        configuration.
  \item \textbf{Conclusions \& Future Work}.
\end{enumerate}
\newpage

\section*{Part B --- The core exhibits (each answers one question with one number)}
"""

doc += exhibit(r"Exhibit 1 --- Cross-engine trade-off: \textsc{Radix}",
    r"How do the four engines trade runtime vs.\ cycle and vs.\ permutations as the base shrinks (BvN$\to$B2)?",
    r"Engines differ ${>}20\times$ in $N$ at BvN yet all converge to $N\approx657$ at $B{=}2$; runtime drops up to $60\times$ for $+38\%$ cycle.",
    r"\includegraphics[width=0.49\textwidth]{plots_pdf/tradeoff_runtime_cycle.pdf}\hfill\includegraphics[width=0.49\textwidth]{plots_pdf/tradeoff_runtime_permutations.pdf}")
doc += exhibit(r"Exhibit 2 --- Cross-engine trade-off: \textsc{Euler} (end-to-end runtime, depths 1--4)",
    r"Does splitting pay end-to-end (split $+$ max-leaf), and what does it cost per engine?",
    r"GW Dynamic total $1330\to109$\,ms ($12\times$) at $C=S$ unchanged; price: $N$ $1122\to3268$; GW Static's $N$ stays flat ($\bar\lambda\approx1$ saturated).",
    r"\includegraphics[width=\textwidth]{plots_pdf/euler_tradeoff_corrected.pdf}")
doc += exhibit(r"Exhibit 3 --- Winner map, sensible regime only ($r\geq5$, $T_{config}$: 1\,$\mu$s--1\,ms)",
    r"Which configuration minimises DCT across realistic hardware, restricted to the transmission-dominated regime?",
    r"Radix nearly vanishes (3\%): the choice reduces to \emph{which Euler depth}, set by $T_{config}$ --- d4 (66\%) below ${\sim}0.15$\,ms, then d3 (21\%), d2 (10\%). The ratio $r$ barely matters in this regime.",
    r"\includegraphics[width=\textwidth]{plots_pdf/dct_param_space_ratio.pdf}")
doc += exhibit(r"Exhibit 4 --- Winner map as 3-D DCT surfaces",
    r"What does the completion-time landscape look like, and where do the two dominant configurations cross?",
    r"The surfaces' intersection curve \emph{is} the crossover boundary; the minimum-DCT floor is coloured by winner.",
    r"\includegraphics[width=\textwidth]{plots_pdf/dct_3d_surfaces.pdf}")
doc += exhibit(r"Exhibit 5 --- Table: all engines across \textsc{Radix} (dense summary)",
    r"What does each engine + Radix base cost in runtime, cycle, permutations?",
    r"All strong engines hit $C=S=8211$ at BvN; Radix $B{=}2$ cuts Hungarian runtime $641\to69$\,ms for $+38\%$ cycle.",
    "\\begin{center}\\small\n" + t_dense + "\n\\end{center}")
doc += exhibit(r"Exhibit 6 --- Table: all engines across \textsc{Euler}",
    r"How do extraction time, cycle, and permutations evolve with split depth for every engine?",
    r"The three strong engines hold $C=S=8174$ at every depth; extraction falls ${\approx}2\times$ per level (split $<50$\,ms at depth 3).",
    "\\begin{center}\\small\n" + t_euler + "\n\\end{center}")
doc += exhibit(r"Exhibit 7 --- DCT vs.\ $T_{config}$",
    r"When does the decomposition choice change completion time on real optical hardware ($T_{unit}=0.01$\,ms)?",
    r"Hungarian Euler d4 is lowest for $T_{config}\lesssim80\,\mu$s; Radix $B{=}2$ takes over beyond it.",
    r"\begin{center}\includegraphics[width=0.85\textwidth]{plots_pdf/dct_vs_tconfig.pdf}\end{center}")
doc += exhibit(r"Exhibit 8 --- Table: Euler vs.\ Radix (Hungarian matching on every unit)",
    r"Head-to-head at comparable parallel-unit counts, split time charged: who wins on Total, at what cycle cost?",
    r"Euler d1 Total $320$\,ms vs Radix $B8$ $303$\,ms --- competitive, but Euler keeps $C=S$ ($C/S=1.0$) vs Radix's $+20\%$.",
    "\\begin{center}\\small\n" + t_evr + "\n\\end{center}")
doc += exhibit(r"Exhibit 9 --- Design space $(N, C, T_{calc})$, total runtime",
    r"How do the two frameworks move through the full three-metric cost space that DCT prices?",
    r"Radix and Euler leave each engine's BvN anchor in \emph{orthogonal} directions; no move improves all three metrics at once.",
    r"\includegraphics[width=\textwidth]{plots_pdf/design_space_projections.pdf}")
doc += exhibit(r"Exhibit 10 --- Winner map at realistic hardware; sensitivity to $T_{calc}$ (per last meeting)",
    r"$T_{config}$ restricted to realistic values (1\,$\mu$s--1\,ms), ratio $r\geq5$: who wins with full computation time, half, and none (prescheduled)?",
    r"With full $T_{calc}$, deep Hungarian Euler wins ${\sim}97\%$ of the map; halving $T_{calc}$ barely moves the boundaries; \textbf{at $T_{calc}=0$ plain Hungarian BvN wins 100\%} --- every framework exists only to buy computation time.",
    r"\includegraphics[width=\textwidth]{plots_pdf/winner_map_tcalc_sensitivity.pdf}")

doc += exhibit(r"Exhibit 11 --- FULL grid: every engine $\times$ every framework $\times$ every step strategy",
    r"How does the step strategy (min / median / max $\lambda$ per peel) interact with all engines and frameworks? (dense, 105 matrices; Euler rt $=$ split$+$max-leaf)",
    r"(i) Hungarian BvN$+$max reaches $N{=}175$ --- $\bar\lambda\approx61$, nearly the $W_{\max}{=}64$ floor; (ii) median cuts WFA BvN runtime $27\times$ (1700$\to$62\,ms); (iii) strategies $\neq$ min break the $C{=}S$ guarantee everywhere (Euler included); (iv) at $B{=}2$ all strategies coincide (0/1 planes) --- a built-in sanity check.",
    r"{\footnotesize \input{strategy_full_table.tex}}")
import pandas as pd
_tt = pd.read_csv("C:/Users/ofekc/Desktop/Msc/Thesis/bvn_project/run/ttfp/ttfp_results.csv")
_pv = _tt.pivot_table(index="framework", columns="engine", values="ttfp_ms", aggfunc="mean")
_order = ["BvN", "Radix B2", "Radix B4", "Radix B8", "Radix B16",
          "Euler d1", "Euler d2", "Euler d3", "Euler d4"]
_rows = "\n".join(
    f"{fr} & " + " & ".join(f"{_pv.loc[fr, e]:.2f}" for e in ["Hungarian", "GW (dyn=stat)", "WFA"]) + r" \\"
    for fr in _order)
ttfp_tab = (r"\begin{tabular}{l rrr}" "\n" r"\toprule" "\n"
            r"Framework & Hungarian & GW (dyn$=$stat) & WFA \\" "\n" r"\midrule" "\n"
            + _rows + "\n" + r"\bottomrule" "\n" r"\end{tabular}")
doc += exhibit(r"Exhibit 12 --- Time to first configuration (TTFP)",
    r"How long until the switch holds its \emph{first} usable configuration? (TTFP $=$ required preprocessing $+$ one matching extraction; Radix $=$ digit extraction $+$ fastest plane; Euler $=$ split wall $+$ fastest leaf; 30 dense matrices)",
    r"The total-runtime ranking \emph{reverses}: BvN/Radix deliver the first configuration in ${\sim}1$\,ms (WFA BvN: $0.11$\,ms) while Euler needs $20$--$41$\,ms --- its split is a hard barrier. BvN/Radix can pipeline (transmit while computing); Euler cannot start until the split ends.",
    "\\begin{center}\\small\n" + ttfp_tab + "\n\\end{center}")

doc += r"""
\section*{Part C --- Theoretical directions}

\subsection*{C.1 Claims we can already state as theorems}
\begin{enumerate}[label=\textbf{T\arabic*.},itemsep=3pt]
  \item \textbf{Strong $\Leftrightarrow$ optimal cycle.} A decomposition satisfies $C=S$ iff every
        configuration is a \emph{full} permutation. On an $S$-regular matrix, peeling a permutation
        preserves regularity, so any engine with BFS augmenting paths achieves $C=S$ all the way down
        (K\H{o}nig--Hall). \emph{This is the formal ``algorithmic superiority'' of the strong engines:
        Hungarian / GW Dynamic / GW Static provably attain the cycle-length optimum; WFA provably need not.}
  \item \textbf{Configuration lower bound (weight cap).} With integer entries $\le W_{\max}$:
        $N \ge \lceil C/W_{\max}\rceil \ge \lceil S/W_{\max}\rceil$. For our benchmark: floor $=129$;
        heuristics deliver $477$--$657$, i.e.\ $4$--$10\times$ above the floor --- a quantified gap.
  \item \textbf{Configuration lower bound (support).} Each configuration covers at most one cell per
        row, so $N \ge \max_i \mathrm{nnz}(\text{row } i)$. Combined with T2:
        $N \ge \max\{\lceil S/W_{\max}\rceil,\ \max_i \mathrm{nnz}_i\}$.
  \item \textbf{Optimum structure \& designer formula.} DCT is linear over $(N, C, T_{calc})$, so the
        optimum is always a vertex of the lower convex hull of achievable points; for any two
        configurations the break-even is closed-form,
        $T_{unit}^{*} = \frac{(T_{calc}^{B}-T_{calc}^{A}) + (N_B-N_A)\,T_{config}}{C_A - C_B}$ ---
        this \emph{is} the boundary curve in every winner map we measured.
  \item \textbf{The space is intractably large.} Counting perfect matchings is the permanent
        (\#P-complete, Valiant 1979); for the unit $K_{n,n}$ case strong decompositions biject with
        Latin squares (super-exponential). Even a $4{\times}4$, $K{=}3$ toy admits $48{,}840$
        decompositions (verified by brute force against the edge-colouring count). Exhaustive search
        is hopeless $\Rightarrow$ heuristics + navigation results are the right currency.
  \item \textbf{Euler split preserves optimality.} An Euler-tour 2-colouring of an $S$-regular
        multigraph yields two $(S/2)$-regular halves; by induction every leaf is regular, so strong
        engines keep $C=S$ at \emph{every} depth. \emph{Superiority claim: Euler is the only framework
        of the three that reduces compute time while provably preserving the cycle-length optimum.}
\end{enumerate}

\subsection*{C.2 Conjectures we believe are true (evidence strong, proof open)}
\begin{enumerate}[label=\textbf{C\arabic*.},itemsep=3pt]
  \item \textbf{Frontier collapse / dominated bases.} For random $K$-regular matrices, the
        intermediate Radix bases ($B=4,8,16$) are convex-dominated by $\{$BvN-or-Euler, $B{=}2\}$ ---
        they are \emph{never} DCT-optimal for any hardware. (Empirically: 100\% of our winner maps.)
        Proof route: expectation formulas for $N(B), C(B)$ + convexity in $\log B$.
  \item \textbf{Radix cycle law.} The digit planes $\Delta_d$ are \emph{not} doubly stochastic, so
        per plane only the conservation \emph{lower bound} is guaranteed:
        $C_{\mathrm{Radix}}(B) \ge \sum_d B^d\, S^{*}(\Delta_d)$ where
        $S^{*}(\Delta)=\max(\max_i \mathrm{rowsum}_i, \max_j \mathrm{colsum}_j)$ --- equality holds
        only when the engine attains the optimum on every (non-regular) plane, which is not
        guaranteed. Conjecture, in two parts: (i) strong engines achieve the bound within $o(1)$
        (measured: mean gap $0.5\%$, max $4\%$); (ii) $\mathbb{E}[S^{*}(\Delta_d)]$ concentrates for
        random matrices (balls-in-bins), turning the measured $8$--$54\%$ inflation curve into an
        approximate formula.
  \item \textbf{Weak-engine inflation bound.} WFA's matchings are \emph{maximal}, hence at least half
        the size of maximum ones; conjecture: $C_{\mathrm{WFA}} \le 2S$
        (measured: $1.45S$ dense, $1.61S$ sparse). Proof route: the classic maximal-matching
        $\tfrac12$-approximation, adapted to weighted peeling.
  \item \textbf{Min-$N$ hardness and the gap.} Finding the minimum-$N$ strong decomposition is
        NP-hard (minimum Birkhoff decomposition, Dufoss\'e--U\c{c}ar 2016); our engines sit
        $4$--$10\times$ above the $W_{\max}$ floor. Direction: tighter lower bounds than T2/T3, or an
        approximation guarantee for the greedy max-weight (Hungarian) engine --- is it an
        $O(\log S)$-approximation?
  \item \textbf{Depth scaling law (saturation).} $N(d) \approx 2^d\, N_{\mathrm{single}}(S/2^d)$;
        hence if an engine's step size is unit-bound ($\bar\lambda\to1$, GW Static) then $N(d)$ is
        \emph{flat} in depth, while engines with $\bar\lambda$ growing in density pay
        $N(d)\uparrow$. Explains the measured saturation exactly; provable from a model of
        $\bar\lambda$ vs.\ leaf density.
  \item \textbf{Optimal depth.} On the $r{=}10$ line the optimal Euler depth grows with the compute
        budget but with sharply diminishing returns; conjecture $d^{*} \le \log_2(S/W_{\max})$
        (beyond that, leaves lose the weight structure that makes splitting profitable).
\end{enumerate}

\end{document}
"""
open(f"{PAPER}/meeting_doc.tex", "w", encoding="utf-8").write(doc)
print("meeting_doc.tex written")
