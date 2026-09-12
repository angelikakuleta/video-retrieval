"""Annotation layer shared by the series datasets (TBBT, The Office).

The pipeline of a series dataset is always the same:

1. candidate intervals are written per episode as ``<series>_<episode>_intervals.csv``
   in the format read by ``tools/interval-annotator`` (see :mod:`intervals`),
2. the file is verified by hand in that tool -- rows are removed, times corrected,
   new rows added,
3. the verified file is read back: the masks give the corpus ranges
   (:mod:`ranges`), the annotations update the collective register
   (:mod:`registry`) and become the query files (:mod:`build`).

VATEX does not go through this layer -- its clips are self-contained events and
its adapter lives in :mod:`src.utils.vatex`.
"""
