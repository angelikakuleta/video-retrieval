"""The text side of the threshold measurement (chapter 6.4.3).

Everything here is judged on WORDS: which phrases a query yields, which class
name each of them reaches and how strongly. Nothing in this module touches a
ranking or a Recall, which is what lets it run before the configuration is
frozen.

Three things live here.

The DIAGNOSTIC (:func:`phrase_heads`) documents the reach of the phrase rules
and settles nothing.

The CALIBRATION (:func:`build_threshold_samples`, :func:`threshold_from_judgments`)
fixes the one number the closed-vocabulary signals cannot work without: the
confidence a phrase has to reach before its class counts as found. Its truth is
established by hand -- a sample of distinct (phrase, class) pairs per vocabulary,
judged against the question "is this class a correct counterpart of this phrase
in this sentence?".

One instrument, per vocabulary, and no second one. An earlier design measured a
single threshold against Kinetics and carried it to the others, checking the
carry against hand-written lists of fifteen phrases with a counterpart and
fifteen without; the supervisor replaced that with a judged sample per
vocabulary, and the lists then duplicated the measurement with a worse
instrument. Hand-picked phrases sit on either side of an empty gap in the
confidence distribution, so a threshold read off them lands on the edge of that
gap rather than where precision actually falls -- 0,741 / 0,826 / 1,000 against
the 0,805 / 0,733 / 0,805 the judged samples give.

The OLD PATH (:func:`measure`) took the Kinetics class of a VATEX clip as its
truth: for a clip whose class is one of the 400, the description ought to name
that class. It is kept and reported, but it UNDERSTATES precision and
:func:`literal_agreement` says by how much -- where the phrase is the class name
character for character the match is correct by construction, and the clip label
still disagrees about a third of the time. The thesis shows both numbers and
explains the difference.

Nothing here looks at a ranking or a Recall. Every question is about text, so it
can be answered before the configuration is frozen -- which is the whole reason
the threshold can be settled first and everything else afterwards.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

#: the two series; VATEX is left out on purpose -- its descriptions are what the
#: threshold is measured ON, so counting their phrases would describe the
#: measurement material rather than the material the rules were written for
SERIES = ("tbbt", "office")

#: which phrase kinds the diagnostic covers. The action rule is not among them:
#: its reach is what the threshold measurement itself reports, phrase by phrase.
KINDS = ("object", "expression")

#: heads listed per (kind, dataset); enough to show the shape of the tail
TOP_HEADS = 30


def phrase_heads(
    datasets: tuple[str, ...] = SERIES,
    kinds: tuple[str, ...] = KINDS,
    split: str = "dev",
    top: int | None = TOP_HEADS,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict]]:
    """What the rules pick out of a split's queries, per kind and dataset.

    ``{kind: {dataset: {"heads", "queries_with_phrase", "queries", "phrases"}}}``.

    Two counts, and the difference between them is not bookkeeping. The
    PREDICTED ACTIVATION of a gated signal is the share of QUERIES with at least
    one phrase of its kind, and that is what `tab:aktywacja-dev` carries; the
    number of phrases is larger, because one query often yields two, and it goes
    nowhere. Reporting phrases where the table wants queries overstates the
    activation.

    It documents a LIMIT and settles nothing: the rules are frozen before the
    threshold is measured, so a head that looks wrong here is a sentence in
    chapter 7, not an edit to :mod:`src.retrieval.phrases`.

    The three kinds come off one parse per query, so asking for both costs the
    same as asking for either.
    """
    from src.data.queries import load_queries
    from src.retrieval.phrases import phrases_for

    unknown = sorted(set(kinds) - set(KINDS))
    if unknown:
        raise ValueError(f"phrase kinds outside the diagnostic: {unknown}")

    out: dict[str, dict[str, dict]] = {kind: {} for kind in kinds}
    for dataset in datasets:
        queries = load_queries(dataset, split)
        counted = {kind: Counter() for kind in kinds}
        with_phrase = {kind: 0 for kind in kinds}
        for query in queries:
            found = phrases_for(query)
            for kind in kinds:
                counted[kind].update(phrase.head for phrase in found[kind])
                with_phrase[kind] += bool(found[kind])
        for kind in kinds:
            ordered = sorted(counted[kind].items(), key=lambda item: (-item[1], item[0]))
            phrases = sum(counted[kind].values())
            out[kind][dataset] = {
                "heads": [[head, count] for head, count in ordered[:top]],
                "queries_with_phrase": with_phrase[kind],
                "queries": len(queries),
                "phrases": phrases,
            }
            log(f"{dataset}/{kind}: {with_phrase[kind]} of {len(queries)} queries "
                f"carry one, {phrases} phrases, {len(counted[kind])} distinct heads")
    return out


# -------------------------------------------------------------- the material
#: the base representation whose text encoder the pipeline uses, and therefore
#: the one the threshold has to be measured in. A threshold measured in another
#: space would be a number about a different geometry.
ENCODER = "openclip_vit_h14"

#: precision the threshold is chosen at. Decided before the measurement and not
#: subject to tuning: it is the level at which a match is worth acting on.
PRECISION = 0.9

#: the vocabulary the threshold is measured against
KINETICS = "actions_kinetics400"

#: every vocabulary a threshold is carried to, and which kind of phrase asks it
VOCABULARY_KIND = {
    "objects_yolo11": "object",
    "objects_yoloe_promptfree": "object",
    "expressions_hsemotion": "expression",
    KINETICS: "action",
}

def dev_material(part: str = "dev", encoder: str = ENCODER, *,
                 labelled_only: bool = True) -> list[dict]:
    """Descriptions of the development clips, one per clip.

    The same description the queries carry (``settings.VATEX_EXPERIMENT_DESC``),
    because the threshold has to be measured on the text the pipeline will
    actually be given.

    ``labelled_only`` keeps only the clips whose source class is one of the 400.
    The OLD measurement path (:func:`measure`) needs that, because the clip's
    Kinetics label is the truth it compares against. The CALIBRATION does not,
    and must not have it: its truth is established by hand and does not depend on
    the clip's class, while the filter would cut the material from 616 clips to
    373 and skew what is left. Clips whose class is in Kinetics-400 yield action
    phrases that more often have a real counterpart in that vocabulary, so
    precision at every confidence would come out flattered and the threshold for
    Kinetics too low.

    ``leak_k400`` is not applied either way: the calibration runs no SlowFast, so
    a leaking clip is an ordinary description to it. On dev there is none anyway --
    the draw excluded leaks up front.

    Works on WHAT IS ON DISK, and the count travels with the result, so a
    measurement always says how much material it stood on.
    """
    from src.utils import vatex

    split = vatex.load_split(part=part)
    labels = {str(row["videoID"]):
              (str(row["class_k600"])
               if str(row["class_in_k400_vocab"]).strip().lower() == "yes" else None)
              for _, row in split.iterrows()}

    descriptions = vatex.load_descriptions(vatex.descriptions_file(part))
    chosen = descriptions[descriptions["desc_no"] == vatex.EXPERIMENT_DESC]
    return [{"clip": str(row["videoID"]), "label": labels[str(row["videoID"])],
             "text": str(row["desc_en"]).strip()}
            for _, row in chosen.iterrows()
            if str(row["videoID"]) in labels
            and not (labelled_only and labels[str(row["videoID"])] is None)]


# ----------------------------------------------------------------------- tau
def score_descriptions(material: list[dict], encoder_object, vocabulary,
                       names: list[str], log: Callable[[str], None] = print) -> list[dict]:
    """Every description's best match under BOTH candidate measures.

    A description is one unit of evaluation. Its match is the most confident of
    its action phrases, and it is correct when that phrase's class is the clip's
    class. Measuring per PHRASE instead would count the other verbs of the
    sentence (*use*, *play*, *wear*) as misses no threshold can fix.
    """
    from src.retrieval import vocab_match
    from src.retrieval.phrases import extract
    from src.utils.progress import progress

    report = progress(log, total=len(material))
    out = []
    for row in material:
        phrases = extract(row["text"], "action")
        entry = {"clip": row["clip"], "label": row["label"],
                 "n_phrases": len(phrases), "text": row["text"]}
        for measure in ("top1", "z"):
            matches = (vocab_match.score(phrases, vocabulary, encoder_object,
                                         measure=measure) if phrases else [])
            best = max(matches, key=lambda m: m.confidence, default=None)
            entry[measure] = None if best is None else {
                "confidence": float(best.confidence),
                "class": names[best.class_id],
                "hit": names[best.class_id] == row["label"],
                # kept so that the literal subset can be picked out afterwards:
                # it is what measures the ceiling of this whole path
                "forms": list(best.phrase.forms),
                # how many OTHER phrases of this description matched, and their
                # confidences: the second diagnostic reads them
                "others": sorted((float(m.confidence) for m in matches
                                  if m is not best), reverse=True),
            }
        out.append(entry)
        report(f"{row['clip']}: {len(phrases)} action phrase(s)")
    return out


def tau_from(scored: list[dict], measure: str, precision: float = PRECISION) -> dict:
    """The threshold of one measure, read off the observed confidences.

    Descriptions ordered by confidence; the threshold is the LOWEST confidence at
    which the set above it is still ``precision`` correct. No grid: a grid would
    put the answer on points nobody observed, and the value has to be one a real
    description reached.

    Ties are resolved by value, not by rank -- the pipeline accepts every match
    at or above the threshold, so two descriptions of equal confidence are either
    both in or both out.
    """
    matched = [row[measure] for row in scored if row[measure] is not None]
    total = len(scored)
    if not matched:
        return {"tau": None, "precision_at_tau": None, "coverage": 0.0,
                "n_descriptions": total, "n_matched": 0,
                "precision_without_threshold": None, "reason": "no description matched"}

    ordered = sorted(matched, key=lambda m: m["confidence"], reverse=True)
    best, reachable = None, []
    for value in sorted({m["confidence"] for m in ordered}, reverse=True):
        above = [m for m in ordered if m["confidence"] >= value]
        reached = sum(m["hit"] for m in above) / len(above)
        reachable.append((reached, len(above), value))
        if reached >= precision:
            best = (value, reached, len(above))
    if best is None:
        # There is no threshold at this precision, and that is a RESULT, not a
        # failure to compute: it says the descriptions never name the clip's
        # Kinetics class often enough, at any confidence. The best the curve ever
        # reaches travels with it, so the report can say how far short it fell.
        top_precision, at_n, at_value = max(reachable)
        return {"tau": None, "precision_at_tau": None, "coverage": 0.0,
                "n_descriptions": total, "n_matched": len(matched),
                "precision_without_threshold": sum(m["hit"] for m in matched) / len(matched),
                "best_precision": top_precision, "best_precision_n": at_n,
                "best_precision_confidence": at_value,
                "reason": f"no prefix reaches precision {precision}; the curve peaks "
                          f"at {top_precision:.3f} over {at_n} descriptions"}

    tau, reached, kept = best
    return {"tau": tau, "precision_at_tau": reached,
            "coverage": kept / total if total else 0.0,
            "n_descriptions": total, "n_matched": len(matched), "n_above_tau": kept,
            "precision_without_threshold": sum(m["hit"] for m in matched) / len(matched)}


def literal_agreement(scored: list[dict], measure: str) -> dict:
    """How far the clip label could ever go as a truth, measured on itself.

    Restricted to matches where one FORM of the phrase is the class name
    character for character. There the match is correct by construction -- the
    phrase and the class are the same string -- so every disagreement with the
    clip label is the label's, not the matching's.

    The number that comes out is the CEILING of the old measurement path. It is
    why no threshold at precision 0.9 was reachable there and why the truth is
    now established by hand (:func:`build_threshold_samples`). The old path stays
    in the code and stays in the report, marked as the underestimate it is.
    """
    literal = [m for m in (row[measure] for row in scored)
               if m is not None
               and any(form.strip().lower() == m["class"].strip().lower()
                       for form in m.get("forms", ()))]
    return {"n": len(literal),
            "agreement": (sum(m["hit"] for m in literal) / len(literal))
                         if literal else None}


def choose_measure(scored: list[dict], taus: dict) -> dict:
    """Which confidence measure the pipeline uses, decided by an interval.

    The two measures are compared on COVERAGE at their own thresholds, paired
    per description, with the bootstrap the thesis already uses for VATEX clips.
    An interval covering zero means the data does not separate them, and then
    ``top1`` wins -- it is an absolute similarity and therefore comparable
    between vocabularies of different sizes, which ``z`` is not.

    The rule is written down before the numbers, so it decides them and not the
    other way round.
    """
    from src.evaluation.compare import bootstrap_interval

    if taus["top1"]["tau"] is None or taus["z"]["tau"] is None:
        missing = [m for m in ("top1", "z") if taus[m]["tau"] is None]
        return {"chosen": "top1", "reason": f"no threshold for {missing}; "
                                            "top1 by the default rule",
                "difference": None, "interval": None}

    covered = {}
    for measure in ("top1", "z"):
        threshold = taus[measure]["tau"]
        covered[measure] = [
            1.0 if row[measure] is not None and row[measure]["confidence"] >= threshold
            else 0.0 for row in scored]
    paired = [a - b for a, b in zip(covered["top1"], covered["z"])]
    entry = bootstrap_interval(paired)

    if entry["low"] is None or (entry["low"] <= 0.0 <= entry["high"]):
        chosen = "top1"
        reason = ("the interval of the coverage difference covers zero - top1 by "
                  "the rule fixed beforehand (absolute similarity, comparable "
                  "between vocabularies)")
    else:
        chosen = "top1" if entry["mean"] > 0 else "z"
        reason = f"the interval excludes zero; {chosen} covers more descriptions"
    return {"chosen": chosen, "reason": reason,
            "difference": entry["mean"], "interval": entry}


def diagnostics(scored: list[dict], measure: str, tau: float | None) -> dict:
    """Two numbers that are reported and decide nothing.

    The per-PHRASE precision curve with its ceiling says what the measurement
    would have looked like on the other unit of evaluation, and why that unit was
    rejected: a clip has one label, so at most one phrase of a description can be
    right and the rest are misses no threshold removes.

    The second says how often the pipeline will accept a phrase the calibration
    never looked at -- calibration reads the best phrase of a description, the
    pipeline accepts every phrase above the threshold.
    """
    matched = [row[measure] for row in scored if row[measure] is not None]
    phrases = sum(1 + len(m["others"]) for m in matched)
    # The ceiling is STRUCTURAL, not achieved: a clip carries one label, so at
    # most one phrase of a description can ever be correct. Per-phrase precision
    # is therefore bounded by descriptions/phrases whatever the threshold does --
    # about one over the two action phrases an average description yields.
    ceiling = (len(matched) / phrases) if phrases else None

    curve = []
    if matched:
        for value in sorted({m["confidence"] for m in matched}, reverse=True)[:200]:
            above = [m for m in matched if m["confidence"] >= value]
            # per phrase: the description's own phrases all sit above their own
            # best only when they too clear the value
            kept = sum(1 + sum(1 for other in m["others"] if other >= value)
                       for m in above)
            curve.append([value, (sum(m["hit"] for m in above) / kept) if kept else None])

    second = None
    if tau is not None and matched:
        second = sum(any(other >= tau for other in m["others"])
                     for m in matched if m["confidence"] >= tau)
        above = sum(1 for m in matched if m["confidence"] >= tau)
        second = second / above if above else None
    return {"per_phrase_ceiling": ceiling, "per_phrase_curve": curve,
            "second_phrase_passes": second, "n_phrases": phrases}


# ------------------------- the judged sample: where the truth comes from now
# The clip label cannot serve as the truth -- :func:`literal_agreement` measures
# how far it reaches -- so a person decides instead. For each vocabulary a sample
# of DISTINCT (phrase, top-1 class) pairs is judged by hand, against one
# question: is this class name a correct counterpart of this phrase in this
# sentence?
#
# Distinct pairs, not occurrences: a repeated phrase always draws the same class
# at the same confidence, so judging it twice buys nothing and costs the judge
# time better spent on a pair nobody has seen.
#
# The sample is NOT proportional to the population, and that is deliberate. The
# threshold is looked for where the confidence is high, so that is where the
# judgments have to be: precision is estimated on the ACCEPTED set, and if fewer
# than about 35 judged pairs land above the threshold, the interval around 0.9 is
# wider than +/- 0.10 and the criterion stops meaning anything. Hence 60 pairs
# from the top quartile and 40 from the other three -- the top quartile locates
# the threshold, the rest shows that precision below it really does fall. Both
# halves are re-weighted back to the population when precision is computed.

#: what a literal match scores: one form of the phrase IS the class name, so the
#: two embeddings are the same vector and the cosine is exactly one
EXACT = 1.0

#: pairs judged per vocabulary, and how they split between the strata.
#: Confidence exactly 1.0 is a MASS POINT, not an interval -- for the 4585-name
#: YOLOE catalogue it is most of the pairs, because the catalogue holds the head
#: of nearly every object phrase verbatim. That mass answers ONE question: is a
#: literal match correct? Forty pairs settle it to about +/- 0.09. The rest of
#: the range is what LOCATES the threshold, so it gets the larger half, spread
#: evenly over ten intervals rather than concentrated at the top.
SAMPLE_EXACT = 40
SAMPLE_SPREAD = 60
SAMPLE_BINS = 10
SAMPLE_SIZE = SAMPLE_EXACT + SAMPLE_SPREAD

#: fixed before the draw, like every other seed in this pipeline
SAMPLE_SEED = 1234

#: how close a confidence has to be to 1.0 to count as literal. Not cosmetic:
#: identical embeddings come back as 0.99999994 as often as as 1.0, and without
#: a tolerance float noise alone would decide which pairs join the mass.
TIE = 1e-6

#: below this many judged pairs above the threshold, the interval around the
#: precision level is wider than +/- 0.10 and the criterion says nothing
MIN_ABOVE = 35

#: at more than one in twenty, an unresolved pair stops being noise and becomes
#: a verdict on the question itself (judging rules, tools/judge/README.md)
MAX_UNRESOLVED_SHARE = 0.05

#: the sample file. No truth column and no clip label: a judgment that can see
#: the answer is not a judgment. The confidence is in the file because the
#: threshold is read off it, and the judging tool hides it from the judge.
SAMPLE_COLUMNS = ("vocabulary", "source", "desc_id", "text", "phrase",
                  "matched_class", "confidence")

#: what comes back from the judging tool. Two answers, not three: a pair the
#: judge cannot settle is SKIPPED, which puts it back at the end of the queue and
#: writes nothing -- so one left unresolved to the end is simply absent from the
#: file. Its share is reported against the size of the SAMPLE, which is the only
#: denominator that can see a pair that is not there.
JUDGMENT_COLUMNS = ("vocabulary", "desc_id", "phrase", "matched_class", "correct")
ANSWERS = ("yes", "no")

#: which phrases each vocabulary is sampled with, and from whose text. Only
#: HSEmotion reaches beyond VATEX: facial expression is barely mentioned in clip
#: descriptions, so the series development queries are the rest of what there is.
#: That material will not grow -- the episode selection is frozen -- so the
#: expression threshold stands on a weaker basis than the other three. That is
#: recorded as a limitation, not worked around.
SAMPLE_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "objects_yolo11": ("object", ("vatex",)),
    "objects_yoloe_promptfree": ("object", ("vatex",)),
    KINETICS: ("action", ("vatex",)),
    "expressions_hsemotion": ("expression", ("vatex",) + SERIES),
}

#: Where the sample files and the population they were drawn from are written.
#: NOT under data/interim/<dataset>/: the threshold belongs to no collection. It
#: is a property of the bridge between a query and the class names, it holds for
#: all three collections at once, and its expression sample is drawn from VATEX
#: descriptions AND the development queries of both series -- 21, 9 and 20 of the
#: fifty pairs. A path naming one dataset would contradict the `source` column of
#: the file it points at.
SAMPLE_DIR = Path("data") / "interim" / "threshold"
POPULATION_FILE = "threshold_population.json"

#: where the judged pairs come back, for the same reason and by the same
#: convention. Judged with tools/judge/judge_pairs.py and moved here by hand:
#: no code in this repository writes to data/annotations/ (CLAUDE.md).
JUDGMENTS_FILE = Path("data") / "annotations" / "threshold_judgments.csv"


def sample_texts(dataset: str,
                 material: list[dict]) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """``(source, desc_id, text, characters)`` of one development dataset.

    VATEX contributes the clip descriptions the threshold has always been
    measured on; the series contribute their development queries. A missing
    series query file is a state and not a fault -- those files are rewritten
    after every batch of annotations -- so that dataset drops out of the sample
    and the caller says which one did.
    """
    if dataset == "vatex":
        return [("vatex", row["clip"], row["text"], ()) for row in material]
    from src.data.queries import load_queries

    try:
        queries = load_queries(dataset, "dev")
    except FileNotFoundError:
        return []
    return [(dataset, str(query.event_id), query.desc, tuple(query.identities or ()))
            for query in queries]


def occurrences_of(texts, kind: str, vocabulary, names: list[str], encoder_object,
                   measure: str, log: Callable[[str], None] = print) -> list[dict]:
    """Every phrase of one kind in these texts, with the class it reached.

    One row per OCCURRENCE. Coverage is computed on these -- the question is for
    what share of phrases and of queries the signal switches on, and a phrase
    type occurring forty times is not one fortieth of that answer -- while the
    judging works on the distinct pairs they collapse to.
    """
    from src.retrieval import vocab_match
    from src.retrieval.phrases import extract
    from src.utils.progress import progress

    report = progress(log, total=len(texts))
    rows = []
    for dataset, desc_id, text, characters in texts:
        phrases = extract(text, kind, characters=characters)
        matches = (vocab_match.score(phrases, vocabulary, encoder_object, measure=measure)
                   if phrases else [])
        for phrase, match in zip(phrases, matches):
            rows.append({"source": dataset, "desc_id": desc_id, "text": text,
                         "phrase": phrase.forms[0],
                         "matched_class": names[match.class_id],
                         "confidence": float(match.confidence)})
        report(f"{desc_id}: {len(phrases)} {kind} phrase(s)")
    return rows


def bin_edges(values: Sequence[float]) -> list[float]:
    """The ten equal intervals the sub-literal range is split into.

    Equal in WIDTH, not in population: an interval that turns out empty is a fact
    about the vocabulary worth seeing in the report, and equal-population bins
    would hide it.
    """
    below = [v for v in values if v < EXACT - TIE]
    if not below:
        return []
    low, high = min(below), EXACT - TIE
    if high <= low:
        return []
    step = (high - low) / SAMPLE_BINS
    return [low + i * step for i in range(SAMPLE_BINS + 1)]


def stratum_of(confidence: float, edges: Sequence[float]) -> str:
    """Which stratum a confidence belongs to.

    One function for both sides -- the draw and the precision that re-weights it
    -- so a pair cannot be counted in one stratum and weighted in another. Ties
    stay together by construction: the same value always yields the same name.
    """
    if confidence >= EXACT - TIE:
        return "exact"
    if len(edges) < 2:
        return "all"
    step = (edges[-1] - edges[0]) / SAMPLE_BINS
    if step <= 0:
        return "b01"
    index = int((confidence - edges[0]) / step)
    return f"b{min(SAMPLE_BINS, max(1, index + 1)):02d}"


def allocate(sizes: Sequence[int], total: int) -> list[int]:
    """How many pairs to draw from each interval: as even as the intervals allow.

    An interval holding fewer than its share gives the shortfall back and the
    others take it up, so the sample keeps its size instead of shrinking wherever
    the distribution is thin.
    """
    take = [0] * len(sizes)
    left = total
    while left > 0:
        hungry = [i for i, size in enumerate(sizes) if take[i] < size]
        if not hungry:
            break
        share = max(1, left // len(hungry))
        for i in hungry:
            if left <= 0:
                break
            added = min(share, sizes[i] - take[i], left)
            take[i] += added
            left -= added
    return take


def draw_sample(occurrences: list[dict], size: int = SAMPLE_SIZE,
                seed: int = SAMPLE_SEED) -> tuple[list[dict], dict]:
    """The distinct pairs to judge, stratified by confidence and then shuffled.

    Returns ``(sample, population counts)``. The counts travel with the sample
    because precision has to be re-weighted back to the population: the draw is
    60/40 across strata that are a quarter and three quarters of the pairs, so
    counting the judgments unweighted would report the precision of the sample
    rather than of the vocabulary.

    Row order in the file is random. Ordering by confidence would let the rhythm
    of the answers settle into a pattern the judge could follow.
    """
    import random

    pairs: dict[tuple[str, str], dict] = {}
    for row in sorted(occurrences, key=lambda r: (r["source"], str(r["desc_id"]))):
        key = (row["phrase"], row["matched_class"])
        held = pairs.get(key)
        if held is None:
            pairs[key] = dict(row)
        elif row["confidence"] > held["confidence"]:
            held["confidence"] = row["confidence"]

    ordered = [pairs[key] for key in sorted(pairs)]
    edges = bin_edges([row["confidence"] for row in ordered])
    if len(ordered) <= size or not edges:
        # everything is judged, so there is nothing to stratify and nothing to
        # re-weight: one stratum, of weight one
        edges = []
        strata = {"all": ordered}
        drawn = {"all": list(ordered)}
    else:
        strata = {}
        for row in ordered:
            strata.setdefault(stratum_of(row["confidence"], edges), []).append(row)
        bins = [f"b{i:02d}" for i in range(1, SAMPLE_BINS + 1)]
        want = {"exact": min(SAMPLE_EXACT, len(strata.get("exact", ())))}
        want.update(zip(bins, allocate([len(strata.get(b, ())) for b in bins],
                                       SAMPLE_SPREAD)))
        rng = random.Random(seed)
        drawn = {name: rng.sample(strata[name], want.get(name, 0))
                 for name in sorted(strata)}

    sample = [row for rows in drawn.values() for row in rows]
    random.Random(seed).shuffle(sample)
    return sample, {
        "n_occurrences": len(occurrences), "n_pairs": len(ordered), "edges": edges,
        "strata": {name: len(rows) for name, rows in sorted(strata.items())},
        "sampled": {name: len(rows) for name, rows in sorted(drawn.items())},
    }


def build_threshold_samples(part: str = "dev", encoder: str = ENCODER,
                            measure: str = "top1", out_dir: Path | str = SAMPLE_DIR,
                            log: Callable[[str], None] = print) -> dict:
    """Writes the four files a person judges, one per vocabulary.

    ``threshold_sample_<vocabulary>.csv`` carries the sentence, the phrase, the
    class it reached and the confidence -- and nothing else. There is no truth
    column and no clip label.

    Alongside them goes ``threshold_population.json``: every occurrence the
    sample was drawn from, and how many texts were asked. Coverage is computed
    from it, and so are the stratum weights, so the numbers stay reproducible
    without asking the encoder a second time.
    """
    import csv
    import json

    from src.features import encoders, text_vocab

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # the WHOLE development part: the truth is established by hand and owes
    # nothing to the clip's own class, so filtering on it would only shrink and
    # skew the material (see dev_material)
    material = dev_material(part, encoder, labelled_only=False)
    if not material:
        raise RuntimeError(
            f"no development clip in vatex_split_{part}.csv has a description "
            "on disk - nothing to sample from")
    encoder_object = encoders.build(encoder)

    texts_by_dataset, missing = {}, []
    for dataset in ("vatex",) + SERIES:
        rows = sample_texts(dataset, material)
        texts_by_dataset[dataset] = rows
        if not rows:
            missing.append(dataset)
            log(f"{dataset}: no development queries on disk - not in any sample")

    summary, population = {}, {}
    for vocab_name, (kind, sources) in SAMPLE_SOURCES.items():
        loaded = text_vocab.load(encoder, vocab_name)
        if loaded is None:
            raise RuntimeError(
                f"no cached vocabulary {vocab_name!r} for {encoder!r} - run an "
                "extraction that builds it first (data/cache/vocab/)")
        names, matrix = [str(n) for n in loaded[0]], loaded[1]

        texts = [row for dataset in sources for row in texts_by_dataset[dataset]]
        rows = occurrences_of(texts, kind, matrix, names, encoder_object, measure,
                              log=log)
        sample, counts = draw_sample(rows)

        path = out / f"threshold_sample_{vocab_name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(SAMPLE_COLUMNS)
            for row in sample:
                writer.writerow([vocab_name, row["source"], row["desc_id"], row["text"],
                                 row["phrase"], row["matched_class"],
                                 f"{row['confidence']:.6f}"])

        summary[vocab_name] = {
            "kind": kind, "sources": list(sources),
            "sources_present": [d for d in sources if texts_by_dataset[d]],
            "n_texts": len(texts), "n_sample": len(sample),
            "per_source": dict(Counter(row["source"] for row in sample)),
            "file": str(path), **counts,
        }
        population[vocab_name] = {
            "kind": kind, "n_texts": len(texts), "measure": measure, **counts,
            "occurrences": [[r["source"], r["desc_id"], r["phrase"],
                             r["matched_class"], r["confidence"]] for r in rows],
        }
        log(f"{vocab_name}: {len(rows)} occurrences, {counts['n_pairs']} distinct "
            f"pairs, {len(sample)} to judge -> {path.name}")

    (out / POPULATION_FILE).write_text(
        json.dumps(population, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"samples": summary, "datasets_missing": missing,
            "population_file": str(out / POPULATION_FILE)}


def load_judgments(source) -> list[dict]:
    """The judged pairs, from a CSV path or from anything already iterable."""
    import csv

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            return []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=";")]
    return [dict(row) for row in source]


def coverage_at(entry: dict, threshold: float | None) -> dict:
    """Share of phrase OCCURRENCES and of texts the signal switches on for.

    The denominator of the query share counts every text that was asked,
    including those that yielded no phrase of this kind at all -- there the
    signal is silent too, and pretending otherwise would flatter it.
    """
    occurrences = entry.get("occurrences") or []
    texts = entry.get("n_texts") or 0
    if threshold is None or not occurrences or not texts:
        return {"coverage_phrases": None, "coverage_queries": None,
                "n_occurrences": len(occurrences) or entry.get("n_occurrences"),
                "n_texts": texts or None}
    passing = [row for row in occurrences if row[4] >= threshold]
    return {"coverage_phrases": len(passing) / len(occurrences),
            "coverage_queries": len({(row[0], row[1]) for row in passing}) / texts,
            "n_occurrences": len(occurrences), "n_texts": texts}


def threshold_from_judgments(judgments, vocabulary: str, population=None,
                             precision: float = PRECISION) -> dict:
    """The threshold of one vocabulary, read off the judged pairs.

    The LOWEST confidence at which at least ``precision`` of the accepted pairs
    are correct -- the same tie-safe scan by value as :func:`tau_from`, because
    the pipeline accepts every match at or above the threshold and two pairs of
    equal confidence are either both in or both out.

    Precision is re-weighted by stratum: the draw took 60 pairs from the top
    quarter of the pairs and 40 from the other three quarters, so an unweighted
    count would describe the sample rather than the vocabulary. A stratum's
    weight is its population over the number of pairs actually usable from it,
    which is also how a skipped pair settles: fewer answers in a stratum raise
    that stratum's weight, and nothing else has to be done about it.

    A pair the judge skipped is ABSENT from the answer file, so it is counted
    against the size of the sample rather than against the rows that came back --
    dividing by the rows would report no unresolved pairs however many were
    skipped, because a skip removes the pair from that denominator too.

    Confidences come from the population file, never from the judgment file --
    the judge never saw them and must not be able to move them.

    Returns ``threshold: None`` with a reason instead of raising, so the whole
    path can be exercised before a single pair has been judged.
    """
    import json

    rows = [r for r in load_judgments(judgments)
            if str(r.get("vocabulary", "")).strip() == vocabulary]
    if population is None:
        path = Path(SAMPLE_DIR) / POPULATION_FILE
        population = (json.loads(path.read_text(encoding="utf-8"))
                      if path.exists() else {})
    entry = (population or {}).get(vocabulary, {})

    # the size of the SAMPLE the judging was given, from the file that recorded
    # the draw. Without it there is nothing to measure a skip against.
    drawn = sum((entry.get("sampled") or {}).values()) or None
    empty = {"vocabulary": vocabulary, "threshold": None,
             "precision_at_threshold": None, "n_judged": len(rows),
             "n_sample": drawn, "n_yes": 0, "n_no": 0, "n_unknown": 0,
             "n_unresolved": drawn, "unresolved_share": 1.0 if drawn else None,
             "unresolved_high": None, "n_above": 0, "enough_above": False,
             "strata": {}, "reason": None,
             **coverage_at(entry, None)}
    if not rows:
        return {**empty, "reason": f"no judgments for {vocabulary!r} yet"}

    by_pair: dict[tuple[str, str], float] = {}
    for source, desc_id, phrase, klass, confidence in entry.get("occurrences", []):
        key = (phrase, klass)
        by_pair[key] = max(confidence, by_pair.get(key, confidence))

    edges, strata = entry.get("edges") or [], entry.get("strata") or {}
    if "all" in strata:
        edges = []

    usable, unknown = [], []
    for row in rows:
        key = (str(row.get("phrase", "")).strip(),
               str(row.get("matched_class", "")).strip())
        answer = str(row.get("correct", "")).strip().lower()
        # a pair this population never held, or an answer outside the two the
        # judging can produce: either way the row says nothing about the sample
        if key not in by_pair or answer not in ANSWERS:
            unknown.append(key)
            continue
        usable.append({"confidence": by_pair[key], "correct": answer == "yes",
                       "stratum": stratum_of(by_pair[key], edges)})

    resolved = Counter(row["correct"] for row in usable)
    unresolved = max(0, drawn - len(usable)) if drawn else None
    share = (unresolved / drawn) if drawn else None
    out = {**empty, "n_yes": resolved[True], "n_no": resolved[False],
           "n_unknown": len(unknown), "n_unresolved": unresolved,
           "unresolved_share": share,
           "unresolved_high": None if share is None else share > MAX_UNRESOLVED_SHARE}
    if not usable:
        return {**out, "reason": f"no yes/no judgments for {vocabulary!r} "
                                 f"({len(unknown)} unknown)"}

    judged = Counter(row["stratum"] for row in usable)
    weights = {name: (strata.get(name) or judged[name]) / judged[name]
               for name in judged}
    out["strata"] = {name: {"population": strata.get(name), "judged": judged[name],
                            "weight": weights[name]} for name in judged}

    def precision_above(value):
        above = [row for row in usable if row["confidence"] >= value]
        total = sum(weights[row["stratum"]] for row in above)
        correct = sum(weights[row["stratum"]] for row in above if row["correct"])
        return (correct / total if total else 0.0), len(above)

    best = None
    for value in sorted({row["confidence"] for row in usable}, reverse=True):
        reached, above = precision_above(value)
        if reached >= precision:
            best = (value, reached, above)

    if best is None:
        peak = max((precision_above(v), v) for v in {r["confidence"] for r in usable})
        (reached, above), _ = peak
        return {**out, "reason": f"no prefix reaches precision {precision}; the curve "
                                 f"peaks at {reached:.3f} over {above} judged pairs"}

    threshold, reached, above = best
    return {**out, "threshold": threshold, "precision_at_threshold": reached,
            "n_above": above, "enough_above": above >= MIN_ABOVE,
            **coverage_at(entry, threshold)}


# ------------------- predicted activation, and what fell below the threshold
#: which row of `tab:aktywacja-dev` is asked how. The two "phrases" rows have no
#: threshold to clear -- the open-vocabulary mechanisms answer any phrase of the
#: right kind, which is exactly the coverage difference E5 measures.
ACTIVATION_ROWS = {
    "objects_coco": ("object", "objects_yolo11"),
    "objects_yoloe": ("object", "objects_yoloe_promptfree"),
    "query_time_detection": ("object", None),
    "expressions_hsemotion": ("expression", "expressions_hsemotion"),
    "face_regions": ("expression", None),
    "motion": ("action", KINETICS),
}


def dev_texts(dataset: str, material: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
    """``(text, characters)`` of one development dataset, or an empty list.

    Every dataset reads its query file, VATEX included now that
    `vatex_01_dev.ipynb` has written one. Before it existed the clip descriptions
    stood in, and they still do if it goes away: the query of a clip IS its first
    description (`settings.VATEX_EXPERIMENT_DESC`), so the substitution is exact
    in content and differs only in how many clips it covers.

    A missing query file is a state, not a fault: the series query files are
    rewritten after every batch of annotations, so one of them is simply absent
    from time to time. That dataset drops out of the activation table and the
    rest is still measured -- no step of this plan may demand the full set.
    """
    from src.data.queries import load_queries

    try:
        queries = load_queries(dataset, "dev")
    except FileNotFoundError:
        return [(row["text"], ()) for row in material] if dataset == "vatex" else []
    return [(query.desc, tuple(query.identities or ())) for query in queries]


def activation(datasets: Sequence[str], material: list[dict], vocabularies: dict,
               encoder_object, measure: str, thresholds: dict[str, float],
               log: Callable[[str], None] = print) -> tuple[dict, dict]:
    """Share of QUERIES each signal would speak about, and what fell below.

    Queries, not phrases: a signal is switched on or off once per query, so the
    share of queries is what the activation table carries and what a contribution
    has to be read against. Returns ``(activation, rejected)``.
    """
    from src.retrieval import vocab_match
    from src.retrieval.phrases import extract

    shares: dict[str, dict[str, float]] = {row: {} for row in ACTIVATION_ROWS}
    rejected: dict[str, Counter] = {name: Counter() for name in vocabularies}

    for dataset in datasets:
        texts = dev_texts(dataset, material)
        if not texts:
            log(f"{dataset}: no development queries on disk - activation not measured")
            for row in ACTIVATION_ROWS:
                shares[row][dataset] = None
            continue
        active = {row: 0 for row in ACTIVATION_ROWS}
        for text, characters in texts:
            found = {kind: extract(text, kind, characters=characters)
                     for kind in ("object", "expression", "action")}
            for row, (kind, vocab_name) in ACTIVATION_ROWS.items():
                phrases = found[kind]
                if vocab_name is None:
                    active[row] += bool(phrases)
                    continue
                threshold = thresholds[vocab_name]
                if not phrases or threshold is None:
                    continue        # no threshold yet: the row stays unanswered
                names, matrix = vocabularies[vocab_name]
                matches = vocab_match.score(phrases, matrix, encoder_object,
                                            measure=measure)
                active[row] += any(m.confidence >= threshold for m in matches)
                for phrase, match in zip(phrases, matches):
                    if match.confidence < threshold:
                        rejected[vocab_name][phrase.head] += 1
        for row, (_, vocab_name) in ACTIVATION_ROWS.items():
            # a row whose vocabulary has no threshold was never asked, and zero
            # is not the answer to a question nobody put. It stays unmeasured
            # until the judged threshold exists.
            shares[row][dataset] = (
                None if (vocab_name is not None and thresholds.get(vocab_name) is None)
                else (active[row] / len(texts) if texts else None))
        log(f"{dataset}: {len(texts)} queries scored for activation")

    return shares, {name: [[head, count] for head, count in counter.most_common(10)]
                    for name, counter in rejected.items()}


# ----------------------------------------------------- the whole measurement
def measure(part: str = "dev", encoder: str = ENCODER, datasets=("tbbt", "office", "vatex"),
            judgments=JUDGMENTS_FILE,
            log: Callable[[str], None] = print) -> dict:
    """Everything section 07 asks for, in one payload.

    Runs on WHAT IS ON DISK, and the number of descriptions scored travels with
    the result. What keeps a threshold out of a configuration is not a flag in
    this payload: it is that ``configs/frozen.yaml`` is written by hand and
    committed, and that :func:`src.runners.run._require_matching` refuses a run
    whose closed-vocabulary signals have no block to read.

    The judged thresholds are read HERE and not bolted on afterwards, because
    the activation table is computed at them. The caller used to add the
    ``judged`` block to the returned payload, which is one step too late: by then
    :func:`activation` had already run at the old path's tau -- a single number
    for every vocabulary, and ``None`` since that path stopped producing one --
    so every row of `tab:aktywacja-dev` that depends on a threshold came back
    unanswered. Reading them first is what makes the two tables have a producer.
    """
    from src.features import encoders, text_vocab

    material = dev_material(part, encoder)
    if not material:
        raise RuntimeError(
            f"no development clip has class_in_k400_vocab = yes in "
            f"vatex_split_{part}.csv - nothing to measure the threshold on")
    log(f"material: {len(material)} descriptions of clips with a Kinetics-400 class")

    vocabularies = {}
    for name in VOCABULARY_KIND:
        loaded = text_vocab.load(encoder, name)
        if loaded is None:
            raise RuntimeError(
                f"no cached vocabulary {name!r} for {encoder!r} - run an extraction "
                "that builds it first (data/cache/vocab/)")
        vocabularies[name] = ([str(n) for n in loaded[0]], loaded[1])
    encoder_object = encoders.build(encoder)

    kinetics_names, kinetics_matrix = vocabularies[KINETICS]
    scored = score_descriptions(material, encoder_object, kinetics_matrix,
                                kinetics_names, log=log)
    without_phrase = sum(1 for row in scored if row["n_phrases"] == 0)

    taus = {measure_name: tau_from(scored, measure_name)
            for measure_name in ("top1", "z")}
    choice = choose_measure(scored, taus)
    chosen = choice["chosen"]
    tau = taus[chosen]["tau"]
    log(f"tau({chosen}) = {tau}; {choice['reason']}")

    # Each vocabulary has its OWN threshold, measured on its own judged sample,
    # and that is the value a configuration carries in
    # `matching.threshold_by_vocab`. The activation table has to be computed at
    # those, not at tau, which answers a different question (does a clip's
    # description name the clip's Kinetics class) and answers it with None.
    # A vocabulary nobody has judged yet keeps None here, and
    # `activation` leaves its row unanswered -- which is the honest reading, since
    # zero would mean the question was put and nothing passed.
    judged = {name: threshold_from_judgments(judgments, name)
              for name in VOCABULARY_KIND}
    thresholds = {name: judged[name]["threshold"] for name in vocabularies}
    for name, threshold in thresholds.items():
        log(f"threshold({name}) = {threshold if threshold is not None else '-'}"
            f"  {judged[name].get('reason') or ''}".rstrip())

    shares, rejected = activation(datasets, material, vocabularies, encoder_object,
                                  chosen, thresholds, log=log)

    return {
        "material": {
            "part": part, "encoder": encoder,
            "descriptions": len(material),
            "clips": len({row["clip"] for row in material}),
            "without_action_phrase": without_phrase,
            "without_action_phrase_share": without_phrase / len(material),
        },
        "top1": taus["top1"],
        "z": taus["z"],
        "chosen_measure": chosen,
        "measure_choice": choice,
        "diagnostics": diagnostics(scored, chosen, tau),
        # the ceiling of the clip-label truth, reported next to the threshold it
        # failed to produce: the two together are what the thesis explains
        "literal_agreement": literal_agreement(scored, chosen),
        # the binding thresholds, and the numbers the two tables above stand on
        "judged": judged,
        "activation_dev": shares,
        "rejected_phrases": rejected,
    }

