# Contributing to the catalog

The catalog at https://dragons-project.github.io/grapes-ready-for-picking/ lists public vineyard datasets for 3D scene reconstruction. New datasets and corrections are welcome.

## Submitting a dataset

The catalog is a place for practitioners, in academia and outside it, to share the vineyard data they collect. Submit through the [dataset submission form](https://github.com/DRAGONS-Project/grapes-ready-for-picking/issues/new?template=dataset-submission.yml), or by e-mail to michal.wlodarczyk@ibspan.waw.pl with the subject "Dataset submission: <name>" and the same fields as the form.

The ten characterized datasets of the catalog give a baseline to compare against, and the paper gives a set of recommendations to consider when collecting data.

## Reporting a correction

Use the [correction form](https://github.com/DRAGONS-Project/grapes-ready-for-picking/issues/new?template=correction.yml). Say which dataset, table and column the value is in, the value shown, the correct value, and where the correct value is documented. A correction without a documented source cannot be applied.

## Updating the catalog (maintainers)

The files in `docs/data/` and `docs/text/`, the figure in `docs/static/images/` and the marked regions of `docs/index.html` are generated from the manuscript source by `scripts/catalog/export_bundle.py --src <manuscript source>`. The script converts the LaTeX tables cell by cell and stops on any LaTeX it cannot convert, so edit a value in the source and re-export rather than editing the files by hand. Links for the Paper and Archive buttons go in `docs/text/meta.json` (`links.paper`, `links.archive_doi`); the export keeps them.

## Code

The reconstruction and characterization code is described in [CODE.md](CODE.md). Code contributions follow the usual pull-request process.
