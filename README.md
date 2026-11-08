# Inside the Swarm

Executable supervised multi-agent information-aggregation and causal intervention benchmark. No reinforcement learning or claims of general emergent intelligence. Numerical outputs are actual simulator measurements, not natural-swarm data.

## Run locally

Python 3.10+; CPU is supported (including Apple Silicon); CUDA is selected when available. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.run --mode quick --device cpu
python -m experiments.run --mode standard
python -m experiments.supplement
python paper/build_results.py
python -m experiments.run --mode full
```

Run modes share output paths: preserve or rename `results/` and `checkpoints/` before switching modes. Config mismatch triggers retraining. STANDARD: three seeds, hidden dimension 64, up to 35 epochs × 20 minibatches; FULL: five seeds, up to 70 × 30; QUICK: one seed, dimension 32, five × five. `experiments/run.py:CONFIGS` is the authoritative configuration; `configs/*.json` describes each preset. CPU runtime depends on hardware; the full factorial evaluation is larger than training. Inference batches are capped at 128; mechanistic caches use at most 1024 ten-agent episodes.

Training resumes from `*_latest.pt` after the last completed epoch. `*.complete` marks a finished model; `*.pt` contains best-validation weights, optimizer state, seed and config. An interrupted epoch is repeated with deterministic scenario seeds. Evaluation reruns from the beginning and overwrites aggregate CSVs; models do not need retraining. Never load untrusted PyTorch checkpoint files with `weights_only=False`.

## Kaggle

Upload `notebooks/master_kaggle.ipynb` to Kaggle; it embeds the Python repository and runs top to bottom without a GitHub checkout. Enable Internet for package installation, optionally select a T4, choose MODE, and Run All. Download `inside-the-swarm-results.zip`. To resume, attach prior outputs and copy the saved checkpoint folder to `/kaggle/working/swarm-mech-interp/checkpoints` after extraction and before the run cell. The notebook is regenerated with `python notebooks/build_notebook.py` after source changes. `python notebooks/validate_offline.py` executes its Python cells in QUICK mode with installed dependencies; it skips the pip cell and does not start a Jupyter kernel.

## Model and environment semantics

`y ~ Bernoulli(0.5)`; independently, `P(s_i=y | y,p_i)=p_i`. Reliability is clipped to [0.501,0.99]. The local input is `(2s_i-1,p_i)`; labels are only external supervised targets. Exact log likelihood ratios sum once per unique sensor. The global oracle is an information ceiling; `local_oracle` only sees distinct R-hop sensors. Neither oracle is a learned decentralized controller.

`a[receiver,sender]` masks every message. Shared GRU/message weights recur; each agent retains its own state. Mean aggregation implements GNN; GAT uses scaled dot-product masked attention (not the original additive GAT scorer). Graphs: ring; star; ring plus independent random shortcuts; degree-four ring with second-neighbor shortcut rewiring while retaining a connected backbone (small-world variant). Graphs are fixed within episodes. Training: n=10, random, four rounds, p=.7 ± .1; all other counts and topologies test transfer. Evaluation rounds: 0,1,2,4,8. Noise/reliability and connectivity sweeps are separate and explicitly saved.

Collective probability is the mean of agent probabilities, followed by a 0.5 threshold. This external scoring readout pools global information, so a no-communication MLP can already have high collective accuracy. **Use individual accuracy and local-oracle comparisons to evaluate communication benefits.** Consensus is fraction of agents agreeing with the collective decision; Brier/ECE are computed on collective probabilities. Message cost counts directed edge transmissions × rounds, not bytes. The global oracle's zero message cost means not applicable; it does not claim free distributed access.

## Experiments and data

- E1/E7: `evaluation.csv`, all models × counts × topologies × qualities × rounds; `accuracy_summary.csv` includes seed-level mean, SD and Student-t 95% intervals. `paired_scenarios.csv` retains episode-level paired outputs for the primary condition, and `paired_contrasts.csv` reports seed-level differences.
- E2: `influence.csv`: outgoing zero, cross-episode replacement, and sign-flip interventions. Attention, degree and reliability are descriptive comparators. `influence_controlled.csv` reports descriptive residual variance after controlling degree, reliability and eigenvector centrality. This is not a confirmatory H2 test.
- E3: `activations_*.pt` and `probes.csv`: held-out episode splits, fixed Ridge regularization, private log likelihood ratio, unique radius-limited accumulated evidence, local posterior entropy, final consensus agreement, and final predicted collective label. Permuted-episode targets are negative controls. Probe R² establishes decodability, not causality.
- E4/E5: `circuits.csv`, `edges.csv`, `layers.csv`, `ablations.csv`. Discovery seeds 400000+s rank channels independently of evaluation; random and nearest message-magnitude controls have equal channel counts. Edge ranking uses a fixed ring; edge ablation is directed. Hidden ablations act before recurrent updates, not directly on the final readout. No unique/minimal circuit is claimed.
- E6: `patching.csv`: one sensor flipped, other inputs/graph fixed; patch clean states into corrupted executions at each recurrence. Identity, shuffled-clean, and another-node controls. Signed final logit recovery and recovery among clean-correct/corrupt-wrong decisions are saved. NaN means no damaged decisions; never replace it with zero. A patch at t=0 restoring the flipped sensor is an expected positive control.
- E8: `robustness.csv`: outgoing-message node failures, edge removal, and message noise. Targeted node positions are selected on independent discovery scenarios. Run the supplemental pass to replace initial edge-removal results with per-episode sampling of existing undirected edges; `realized_edge_removal` records rounding effects. It also exports circuit transfer across counts/topologies, graph diameter/radius coverage, generalization gaps, gradient × activation attribution, and immediate/downstream patch diagnostics.

All ten PNG figures are generated at 350 DPI; `results/figures/F*_data.csv` supplies their data. F01 alone is a schematic. CIs use independent training seeds after condition aggregation; individual agents are never treated as replicates. With three seeds intervals are low-powered and can extend outside [0,1]. No significance claim is automated. Training/validation/test scenario streams are distinct; evaluation scenarios are paired across methods. Determinism is tested on CPU; bit-identical results across PyTorch versions/devices are not promised.

## Paper

`paper/main.tex` is an arXiv-style draft with verified references and all ten figure inclusions. From `paper/`, compile with `pdflatex main.tex` twice, or open it in the Codex LaTeX editor. It is not submission-ready. Hypotheses remain unverified unless supported by completed analyses. The scope is a 72-hour research sprint implementation, not evidence of general collective intelligence.

## Completed local run

STANDARD completed on CPU with three seeds, nine best-validation model checkpoints and 5,760 evaluation rows; all ten figures have CSV data and 350 DPI PNGs. The 11-test suite and offline QUICK notebook code-cell execution passed. The manuscript compiled in the Codex desktop compiler. Kaggle frontend, package installation, CUDA execution, and FULL mode have not been executed here. See `results/manifest.json`, `results/notebook_validation.json`, `results/failures.json`, and `results/hypothesis_status.json` for provenance and limitations.

At the primary condition (N=10, random graph, reliability .75, four rounds), GNN individual accuracy was .9501 versus MLP .7472. Collective GNN accuracy was .9557 versus MLP .9564; the paired difference CI includes zero. These values come from the saved STANDARD run and do not claim broad emergent intelligence.

# insidetheswarm
