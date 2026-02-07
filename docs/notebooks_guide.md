# Notebooks Guide

This guide maps each notebook to its purpose and the evidence it produces.

## 01_quickstart_simulation.ipynb

Purpose:
- Run a baseline configuration in a few lines.
- Inspect shapes and plot a snapshot.

Use when:
- You want a quick sanity check or a minimal demo.

## 02_validation_v1_v2_v3.ipynb

Purpose:
- Run the three validation scripts and produce the figures used in the report.

Use when:
- You want visual confirmation that the boundary conditions and regimes behave as expected.

## 03_convergence_and_1d_benchmark.ipynb

Purpose:
- Run grid refinement and the analytic 1D benchmark.
- Load the reports and inspect summary values.

Use when:
- You need numerical evidence that the solver converges and matches an analytic case.

## 04_literature_compare_lidocaine.ipynb

Purpose:
- Compare simulated permeability and lag time to published values.

Use when:
- You want to show the model can match literature targets.

## 05_dataset_pipeline.ipynb

Purpose:
- Build a tiny dataset and inspect the processed splits.

Use when:
- You want a compact example of the ML data pipeline.
