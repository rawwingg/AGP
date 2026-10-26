# AGP design notes — LaTeX sources

Two design documents and a technical report that formalise the reinforcement-learning setup for the Art Gallery
Problem, plus everything needed to rebuild them.

| File | What it is |
| --- | --- |
| `agp_mdp_formulation.tex` / `.pdf` | **Document 1.** The problem written as a Markov Decision Process: what the agent sees, what it can do, how the world changes, and what it gets paid. 17 pages. |
| `agp_staged_ppo_curriculum.tex` / `.pdf` | **Document 2.** The training plan on top of that MDP: three stages (saturate → prune → refine), the reward-weight schedule, PPO settings, ablations, and an implementation roadmap. 17 pages. |
| `agp_paper.tex` / `.pdf` | **Technical report.** Explains every current design decision in one place: the geometry behind the two-stage curriculum, the spatial policy network and its factored continuous action, and the MDP and reward with the exact constants from `agp/`. Seven TikZ/pgfplots figures, no external image files. 16 pages. Upload `docs/agp_paper_overleaf.zip` to Overleaf to edit it there. |
| `agpstyle.sty` | Shared preamble: packages, colours, theorem environments, and all the maths shorthands (`\Free`, `\Vis`, `\ADD`, `\stageI`, …). Edit here, not in the documents. |
| `refs.bib` | Shared bibliography. ~30 entries: the three locomotion papers, Art Gallery theory, and the RL algorithm literature. |
| `Makefile` | Build rules. |

## Building

Needs [Tectonic](https://tectonic-typesetting.github.io/), a self-contained LaTeX engine
that downloads packages on demand:

```sh
brew install tectonic      # one-off
cd docs/latex
make                       # builds all three PDFs
make clean                 # remove .aux/.log/.bbl etc., keep the PDFs
make purge                 # remove the PDFs too
```

The first build needs a network connection (Tectonic fetches the TeX Live bundle, ~2–5
minutes); after that it is cached and a full rebuild takes a couple of seconds. Tectonic
runs BibTeX and the repeated passes automatically, so there is no `latexmk` dance.

A normal TeX Live install works too — `latexmk -pdf agp_mdp_formulation.tex` — but you
will need `natbib`, `cleveref`, `tabularx`, `booktabs`, `algorithm2e`'s `algorithm` +
`algpseudocode`, `tikz`, and `microtype`.

## Conventions used in the sources

- **Everything shared lives in `agpstyle.sty`.** Notation is defined once as a macro
  (`\Free` for the free space, `\Vis` for the visibility polygon, `\Guards` for the guard
  set, `\ADD`/`\DEL`/`\STOP` for action modes, `\stageI`/`\stageII`/`\stageIII` for stage
  names). If a symbol needs to change, change the macro and both documents follow.
- **`\code{...}`** for filenames, identifiers and config constants.
- **`\takeaway{...}`** for the boxed design notes.
- **Cross-references use `\cref`**, which supplies the word ("section 5.5", "Proposition
  6") automatically.
- **References *between* the two documents are written as prose**, not `\cref`, because
  each document has to compile on its own. If you add a new cross-document pointer, name
  the result rather than its number (e.g. "Document 1's telescoping proposition").
- `\sloppy` and a generous `\emergencystretch` are on, because long code identifiers
  cannot be hyphenated and would otherwise overflow the margin. The cost is occasional
  loose inter-word spacing.

## Source material

The three locomotion papers these documents draw on are in
[`../../references/locomotion/`](../../references/locomotion/), with a README explaining
what was taken from each.

The built PDFs are committed one level up, in [`../`](../); `make` copies them there.
The code that implements these two documents is in [`../../agp/`](../../agp/).
