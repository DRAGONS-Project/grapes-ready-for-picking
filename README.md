# Grapes Ready for Picking

*Grapes Ready for Picking: Investigating the Readiness of Precision Viticulture Datasets for Scene Reconstruction*

A living catalog of public vineyard datasets for 3D scene reconstruction, and the code of the paper that built it. For each dataset the catalog records what was captured and how, and for the ten that contain overlapping views it gives a readiness class for reconstruction: ripe, ripening, or unripe.

**Catalog:** https://dragons-project.github.io/grapes-ready-for-picking/

## Submitting a dataset or a correction

- **New dataset:** use the [dataset submission form](https://github.com/DRAGONS-Project/grapes-ready-for-picking/issues/new?template=dataset-submission.yml), or e-mail michal.wlodarczyk@ibspan.waw.pl.
- **Correction:** use the [correction form](https://github.com/DRAGONS-Project/grapes-ready-for-picking/issues/new?template=correction.yml).

The catalog is a place for practitioners, in academia and outside it, to share the vineyard data they collect. The ten characterized datasets give a baseline to compare against, and the paper gives a set of recommendations to consider when collecting data. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Code

The repository holds the code behind the catalog: frame sampling, structure from motion (pycolmap) and 3D Gaussian Splatting (gsplat), the characterization of each dataset under one fixed configuration, the preprocessing tested on datasets that fail to reconstruct, and SLURM launchers to run it on a cluster.
Setup and usage are in [CODE.md](CODE.md).

## How to cite

The paper is not published yet.
A citation will be available here once it is.

## Licence

Code: MIT ([LICENSE](LICENSE)). Catalog data: to be confirmed.
