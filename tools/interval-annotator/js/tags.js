/* Tag vocabulary.

   Tags are free-form labels on a row - the tool has no opinion about what any
   of them means, so the same build works for "wymaga_osoby", "reshoot" or
   anything else. Every tag that has been used is remembered in localStorage
   and offered as a suggestion afterwards; the panel in the header is the place
   to add, rename and drop them.

   The vocabulary is global (not per video): it is the point of remembering it. */

const TAGS_KEY = "interval-annotator:tags";

const tags = {
  known: [],

  /* One tag is one token: no spaces, no ';' or ',' (both are CSV separators). */
  clean(name) {
    return String(name || "")
      .trim()
      .replace(/[;,"]/g, "")
      .replace(/\s+/g, "_")
      .replace(/^_+|_+$/g, "");
  },

  index(name) {
    const key = this.clean(name).toLowerCase();
    return this.known.findIndex(t => t.toLowerCase() === key);
  },

  has(name) {
    return this.index(name) >= 0;
  },

  load() {
    try {
      const raw = localStorage.getItem(TAGS_KEY);
      const list = raw ? JSON.parse(raw) : [];
      this.known = Array.isArray(list)
        ? list.map(t => this.clean(t)).filter(Boolean)
        : [];
    } catch (e) {
      this.known = [];
    }
    this.sort();
    return this.known;
  },

  save() {
    try {
      localStorage.setItem(TAGS_KEY, JSON.stringify(this.known));
      return true;
    } catch (e) {
      return false;
    }
  },

  sort() {
    this.known.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  },

  /* Adds a tag to the vocabulary. Returns the stored spelling, or "" when the
     name was empty. An existing tag keeps the spelling it already has. */
  add(name) {
    const clean = this.clean(name);
    if (!clean) return "";
    const at = this.index(clean);
    if (at >= 0) return this.known[at];
    this.known.push(clean);
    this.sort();
    this.save();
    return clean;
  },

  /* Everything seen on an imported or restored row goes into the vocabulary. */
  remember(list) {
    let changed = false;
    for (const name of list || []) {
      const clean = this.clean(name);
      if (clean && !this.has(clean)) { this.known.push(clean); changed = true; }
    }
    if (changed) { this.sort(); this.save(); }
  },

  /* Drops the tag from the vocabulary. Rows are handled by the caller. */
  forget(name) {
    const at = this.index(name);
    if (at < 0) return false;
    this.known.splice(at, 1);
    this.save();
    return true;
  },

  rename(from, to) {
    const clean = this.clean(to);
    if (!clean) return "";
    const at = this.index(from);
    if (at < 0) return "";
    const collision = this.index(clean);
    if (collision >= 0 && collision !== at) this.known.splice(at, 1);
    else this.known[at] = clean;
    this.sort();
    this.save();
    return clean;
  }
};

/* Row helpers - a row's tags are a plain array of strings, in the order added. */
const tagsOf = item => (Array.isArray(item.tags) ? item.tags : []);

function addTagTo(item, name) {
  const clean = tags.clean(name);
  if (!clean) return "";
  item.tags = tagsOf(item);
  const at = item.tags.findIndex(t => t.toLowerCase() === clean.toLowerCase());
  if (at >= 0) return item.tags[at];
  const stored = tags.add(clean);
  item.tags.push(stored);
  return stored;
}

function removeTagFrom(item, name) {
  const key = tags.clean(name).toLowerCase();
  item.tags = tagsOf(item).filter(t => t.toLowerCase() !== key);
}

/* How many rows carry the tag - shown in the panel before dropping it. */
function tagUsage(items, name) {
  const key = tags.clean(name).toLowerCase();
  return items.filter(i => tagsOf(i).some(t => t.toLowerCase() === key)).length;
}

tags.load();
