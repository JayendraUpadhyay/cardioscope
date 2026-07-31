# CardioScope — Coder Workspace Template

This Terraform template provisions a ready-to-use Coder cloud development
environment for the CardioScope project. It clones the repository, sets up
the Python environment (backend/notebooks) and Node environment (frontend),
and exposes an interactive parameter form so anyone (including hackathon
judges) can spin up a fully working CardioScope dev environment in one click
— no local setup, no dependency wrangling.

## What this fulfills

Hack4Health's Coder sponsor requirement: using Coder's product suite to make
the project's research reproducible and easy to explore, in lieu of a static
paper-only submission.

## Interactive parameters exposed in the Coder workspace creation form

- **Include GPU?** (bool) — if enabled, provisions a GPU-backed workspace for
  faster ECG autoencoder retraining; defaults to false since the trained
  model artifacts are already committed to the repo and GPU is not required
  to just run the demo.
- **Component to launch on start** (dropdown) — choose which part of the
  project to boot automatically: `full-stack` (backend + frontend together),
  `backend-only`, `frontend-only`, or `notebooks-only` (JupyterLab with the
  6 analysis notebooks open).
- **Dataset subset to preload** (dropdown) — `sample` (small bundled sample
  CSVs, fast startup) or `full` (pulls the full datasets from the repo's
  `data/` directory if present) — kept separate because the raw ECG CSV is
  large and not everyone needs it just to explore the code/notebooks.
- **Workspace region** (dropdown) — standard Coder region selector.

## Files

- `main.tf` — the Coder/Terraform template defining the parameters above and
  the workspace provisioning logic (installs Python + Node deps, clones the
  repo, starts the chosen component).
- `startup.sh` — helper script referenced by `main.tf` that performs the
  actual environment setup based on the selected parameters.

## How to use this (for the student / submission)

1. Create a free Coder deployment (or use the hackathon-provided Coder
   instance, per the Discord guide) at https://coder.com.
2. Push this `coder/` directory's template to your Coder deployment as a new
   template (`coder templates push cardioscope`), or upload `main.tf` via the
   Coder dashboard's "New Template" flow.
3. Once published, share the generated Coder template URL in your Kaggle
   Writeup / README under a "Try it in Coder" section, alongside the Vercel
   demo link and GitHub repo — this is the artifact that satisfies the
   sponsor integration requirement.
4. Take a screenshot of the workspace creation form (showing the interactive
   parameters above) and include it in your written report's methodology or
   appendix section, per the sponsor's guidance to make integration explicit
   for judges rather than assuming they'll dig through code to find it.
