/* The model: one flat list of {id, type, start, end, description, tags}.
   type "a" = annotation (exported as "event"), "m" = mask (intro, credits -
   excluded from processing; its description is optional).
   tags is a free-form list of labels; the tool gives none of them a meaning.
   Rows are identified by their stable id, never by position, so editing one
   row can never move or overwrite another. Sorting happens on request only. */

const store = {
  items: [],
  activeId: null,
  videoName: "",

  find(id) {
    return this.items.find(i => i.id === id);
  },

  of(type) {
    return this.items.filter(i => i.type === type);
  },

  /* Annotations: "<episode>_001". Masks: "<episode>_m001". */
  nextId(prefix, type) {
    const base = (prefix || "clip").trim() + "_" + (type === "m" ? "m" : "");
    const taken = new Set(this.items.map(i => i.id));
    let n = 1;
    while (taken.has(base + String(n).padStart(3, "0"))) n++;
    return base + String(n).padStart(3, "0");
  },

  /* Import. An id that came with the file is kept exactly as it is - for
     annotations imported from an external source it carries the id of that
     source (e.g. "tbbt_s01e01_d97650" holds the TVR desc_id 97650), and that
     is what tells us afterwards which rows were removed by hand. A renamed id
     would break that link, so a row whose id is already taken is SKIPPED and
     reported instead of being given a suffix. Only a row that arrives without
     an id gets one minted here. */
  merge(items, prefix) {
    const skipped = [];
    for (const item of items) {
      if (!Array.isArray(item.tags)) item.tags = [];
      if (!item.id) item.id = this.nextId(prefix, item.type);
      else if (this.find(item.id)) { skipped.push(item.id); continue; }
      this.items.push(item);
      tags.remember(item.tags);
    }
    return { added: items.length - skipped.length, skipped };
  },

  /* Ids that appear more than once, and rows with no id at all. */
  idProblems() {
    const seen = new Set(), duplicates = new Set();
    let missing = 0;
    for (const item of this.items) {
      if (!item.id) { missing++; continue; }
      if (seen.has(item.id)) duplicates.add(item.id);
      seen.add(item.id);
    }
    return { duplicates: [...duplicates], missing };
  },

  add(item) {
    this.items.push(item);
    this.activeId = item.id;
  },

  remove(id) {
    this.items = this.items.filter(i => i.id !== id);
    if (this.activeId === id) this.activeId = null;
  },

  /* Masks first at equal start, so the list reads intro -> content. */
  sorted() {
    return this.items.slice().sort((a, b) => a.start - b.start || a.end - b.end);
  },

  sortByStart() {
    this.items = this.sorted();
  },

  /* Annotations may overlap each other - reported for information only. */
  overlaps() {
    const annotations = this.of("a");
    const ids = new Set();
    let pairs = 0;
    for (let i = 0; i < annotations.length; i++)
      for (let j = i + 1; j < annotations.length; j++) {
        const a = annotations[i], b = annotations[j];
        if (a.start < b.end && b.start < a.end) { ids.add(a.id); ids.add(b.id); pairs++; }
      }
    return { ids, pairs };
  },

  /* Annotations must not overlap masks. Returns the ids they clash with. */
  maskClashes(candidate) {
    const others = this.items.filter(i => i.id !== candidate.id && i.type !== candidate.type);
    return others
      .filter(i => candidate.start < i.end && i.start < candidate.end)
      .map(i => i.id);
  },

  /* Masks must not overlap each other. Two masks over the same seconds make
     the content axis ambiguous - the pipeline would have to guess whether the
     shared stretch is cut once or twice - so unlike an annotation on a mask
     this is an error, never a warning. Touching boundaries are fine: the
     intervals are half-open, so they share no frame. */
  maskOverlaps(candidate) {
    if (candidate.type !== "m") return [];
    return this.of("m")
      .filter(i => i.id !== candidate.id
        && candidate.start < i.end && i.start < candidate.end)
      .map(i => i.id);
  },

  /* All pairs of masks that overlap, as "a x b" strings, for the export check. */
  maskOverlapPairs() {
    const masks = this.of("m").slice().sort((a, b) => a.start - b.start);
    const pairs = [];
    for (let i = 0; i < masks.length; i++)
      for (let j = i + 1; j < masks.length; j++)
        if (masks[i].start < masks[j].end && masks[j].start < masks[i].end)
          pairs.push(masks[i].id + " x " + masks[j].id);
    return pairs;
  },

  /* Every id that currently sits on the wrong side of a mask. */
  clashingIds() {
    const ids = new Set();
    for (const item of this.items)
      for (const id of this.maskClashes(item)) { ids.add(id); ids.add(item.id); }
    return ids;
  },

  /* Autosave is a safety net only; the CSV export is the actual save. */
  storageKey(episode) {
    return "interval-annotator:" + (this.videoName || episode || "none");
  },

  save(episode) {
    try {
      localStorage.setItem(this.storageKey(episode), JSON.stringify(this.items));
      return "Autosaved in browser at " + new Date().toLocaleTimeString();
    } catch (e) {
      return "Autosave unavailable - export CSV regularly.";
    }
  },

  loadSaved(episode) {
    try {
      const raw = localStorage.getItem(this.storageKey(episode));
      if (!raw || this.items.length) return false;
      if (!confirm("A browser autosave was found for this video. Load it?")) return false;
      this.items = JSON.parse(raw).map(i => ({ type: "a", tags: [], ...i }));
      this.items.forEach(i => {
        if (!Array.isArray(i.tags)) i.tags = [];
        tags.remember(i.tags);
      });
      return true;
    } catch (e) {
      return false;
    }
  }
};
