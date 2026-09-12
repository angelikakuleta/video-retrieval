#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual judging of phrase-to-class pairs, one keystroke each (chapter 6.4.3).

    python tools/judge/judge_pairs.py \
        --samples data/interim/threshold/threshold_sample_*.csv \
        --output  data/interim/threshold/threshold_judgments.csv

Both arguments are REQUIRED and there are no defaults. What is being judged and
where the answers land are the two things that must never be guessed: a default
nobody typed is a default nobody checked, and a session of manual judging is not
work anyone wants to discover was written somewhere else. Paths are taken as
given, relative to the current directory.

One question per pair: is the matched class a correct equivalent of the phrase in
this sentence? The rules behind that question are in tools/judge/README.md.

Everything the judging needs is a sentence, a phrase and a class name, so this is
a console loop and not an application: no server, no browser, no browser storage.
The answer file is rewritten after EVERY keystroke, so closing the window costs
nothing and there is no separate export step.

The judge must not see the confidence of a match (it would suggest the answer) nor
the clip's source class (that is the ground truth being replaced), so neither is
printed. Both stay in the sample file, because the threshold is computed from them.

Console output is ASCII only: Windows consoles mangle anything else depending on
the code page.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

#: No SAMPLES or OUTPUT constant on purpose: both are required arguments. The
#: script therefore needs no notion of where the repository root is, which is one
#: fewer thing to get wrong when the file moves.

#: input columns produced by scripts/make_threshold_samples.py
SAMPLE_COLUMNS = ("vocabulary", "source", "desc_id", "text", "phrase",
                  "matched_class", "confidence")
OUT_COLUMNS = ("vocabulary", "desc_id", "phrase", "matched_class", "correct")

#: Answers. Case does not matter -- getkey() lowercases. Only two, because the
#: question has two answers; a pair nobody can decide is SKIPPED rather than
#: answered, and an unresolved pair is simply absent from the answer file.
ANSWERS = {"y": "yes", "n": "no"}
KEYS = "[Y]es  [N]o  [S]kip  [P]revious  [Q]uit and save"


def key(row: dict) -> tuple[str, str, str]:
    """Identity of a judgement. Not desc_id: the question is about the PAIR, and
    the same pair asked twice would have to get the same answer."""
    return (row["vocabulary"], row["phrase"], row["matched_class"])


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def write_csv(rows: list[dict], path: Path) -> None:
    """Rewrites the answer file. Atomic, so an interrupted write cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUT_COLUMNS), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def answered(pairs: list[dict], answers: dict, keep: list[dict]) -> list[dict]:
    """Rows to write: everything already in the file that this run is not about,
    followed by the pairs of this run that have an answer.

    ``keep`` is what makes judging one sample file at a time safe. The answer file
    is rewritten whole after every keystroke, so without carrying the other
    vocabularies' rows across, a session spent on one sample would delete the
    answers given to the others -- silently, since nothing would look wrong.
    """
    return keep + [{**{c: r[c] for c in OUT_COLUMNS if c != "correct"},
                    "correct": answers[key(r)]}
                   for r in pairs if key(r) in answers]


def getkey() -> str:
    """One keypress, lowercased. Falls back to a line of input where getch is absent."""
    try:
        import msvcrt
    except ImportError:
        return (sys.stdin.readline().strip() or " ")[:1].lower()
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):      # a function or arrow key: swallow the second byte
        msvcrt.getch()
        return ""
    if ch == b"\x03":                 # Ctrl+C does not raise through getch
        raise KeyboardInterrupt
    if ch == b"\x08":                 # Backspace, same as P
        return "p"
    try:
        return ch.decode("ascii").lower()
    except UnicodeDecodeError:
        return ""


def marked(text: str, phrase: str) -> str:
    """The sentence with the phrase marked. ASCII markers, so no code page can eat them."""
    lowered, needle = text.lower(), phrase.lower()
    at = lowered.find(needle)
    if at < 0:                        # a lemma need not occur verbatim (mugs -> mug)
        head = needle.split()[-1]
        at = lowered.find(head)
        if at < 0:
            return text
        needle = head
    return f"{text[:at]}>>{text[at:at + len(needle)]}<<{text[at + len(needle):]}"


def show(row: dict, done: int, total: int, tally: dict, skipped: int) -> None:
    width = 40
    filled = round(width * done / total) if total else 0
    print("\n" * 2)
    print(f"[{'#' * filled}{'.' * (width - filled)}]  {done + 1} of {total}"
          f"   (yes {tally['yes']} / no {tally['no']}"
          + (f" / skipped {skipped}" if skipped else "") + ")")
    print(f"vocabulary: {row['vocabulary']}   source: {row['source']}")
    print("-" * 76)
    print(f"  {marked(row['text'], row['phrase'])}")
    print()
    print(f"  phrase:  {row['phrase']}")
    print(f"  class:   {row['matched_class']}")
    print("-" * 76)
    print("  Is the class a correct equivalent of the phrase?")
    print(f"  {KEYS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=True, nargs="+", metavar="PATH",
                    help="sample file(s) to judge; a wildcard is expanded here, "
                         "so quote it on shells that do not expand it themselves")
    ap.add_argument("--output", required=True, type=Path, metavar="PATH",
                    help="answer file, rewritten after every keystroke")
    args = ap.parse_args()
    args.output = Path(args.output).expanduser().resolve()

    files = sorted({p for pattern in args.samples
                    for p in (glob.glob(str(Path(pattern).expanduser()))
                              or ([pattern] if Path(pattern).exists() else []))})
    if not files:
        print("No sample file matches:")
        for pattern in args.samples:
            print(f"  {Path(pattern).expanduser().resolve()}")
        return 1

    pairs: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for path in files:
        for row in read_csv(Path(path)):
            missing = [c for c in SAMPLE_COLUMNS if c not in row]
            if missing:
                print(f"{Path(path).name}: missing column(s) {missing}")
                return 1
            if key(row) not in seen:       # the same pair asked once, whatever the file
                seen.add(key(row))
                pairs.append(row)

    existing = read_csv(args.output) if args.output.exists() else []
    mine = {key(r) for r in pairs}
    keep = [r for r in existing if key(r) not in mine]
    answers = {key(r): r["correct"] for r in existing if key(r) in mine}
    tally = {"yes": 0, "no": 0}
    for value in answers.values():
        if value in tally:
            tally[value] += 1

    # Both paths are printed before the first pair. They have defaults so that an
    # outside judge is handed one command rather than two Windows paths to retype,
    # but a default nobody can see is the thing to avoid, not the default itself.
    print("reading:")
    for path in files:
        print(f"  {Path(path)}")
    print(f"writing:\n  {args.output}")
    print(f"\nLoaded {len(pairs)} pairs from {len(files)} file(s).")
    if answers:
        print(f"Already judged: {len(answers)}. Resuming at the first unjudged pair.")

    queue = [r for r in pairs if key(r) not in answers]
    history: list[dict] = [r for r in pairs if key(r) in answers]
    skipped: set[tuple[str, str, str]] = set()
    try:
        while queue:
            if all(key(r) in skipped for r in queue):
                print(f"\n\nOnly skipped pairs are left ({len(queue)}). "
                      "Going through them again -- press Q to stop and leave "
                      "them unresolved.")
                skipped.clear()
            row = queue[0]
            show(row, len(answers), len(pairs), tally, len(skipped))
            answer = None
            while answer is None:
                pressed = getkey()
                if pressed == "q":
                    raise KeyboardInterrupt
                if pressed == "s":                 # defer: back of the queue, no answer
                    skipped.add(key(row))
                    queue.append(queue.pop(0))
                    break
                if pressed == "p":
                    if not history:
                        continue
                    previous = history.pop()
                    tally[answers.pop(key(previous))] -= 1
                    queue.insert(0, previous)
                    break
                answer = ANSWERS.get(pressed)
            if answer is None:                     # came from S or P
                continue
            answers[key(row)] = answer
            tally[answer] += 1
            skipped.discard(key(row))
            history.append(queue.pop(0))
            write_csv(answered(pairs, answers, keep), args.output)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    write_csv(answered(pairs, answers, keep), args.output)
    left = len(pairs) - len(answers)
    print(f"\nSaved {len(answers)} judgements to {args.output}"
          f"   (yes {tally['yes']} / no {tally['no']})")
    if left:
        print(f"{left} pair(s) unresolved. Run again to continue; a pair that "
              "stays unresolved is excluded from the precision and its share "
              "is reported.")
    else:
        print("Complete. Move the file to data/annotations/threshold_judgments.csv, "
              "then run scripts/measure_threshold.py --split dev.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
