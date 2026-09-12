/* DOM wiring: forms, table, keyboard, CSV, autosave. */

const el = id => document.getElementById(id);
const video = el("video");
const fps = () => +el("fps").value || 25;
const masksLocked = () => el("lockMasks").checked;

player.init(video, fps);

/* ---- time readout ------------------------------------------------------ */

function showTime() {
  el("timeSeconds").textContent = seconds(video.currentTime);
  el("timeCode").textContent = timeCode(video.currentTime);
  el("frame").textContent = Math.round(video.currentTime * fps());
}
video.addEventListener("timeupdate", showTime);
video.addEventListener("seeked", showTime);

function setStatus(text) {
  el("status").textContent = text;
}

function autosave() {
  setStatus(store.save(el("episode").value));
}

/* ---- sources ----------------------------------------------------------- */

el("pickVideo").addEventListener("click", () => el("videoFile").click());

el("videoFile").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  store.videoName = file.name;
  el("videoName").textContent = file.name;
  video.src = URL.createObjectURL(file);
  if (!el("episode").value) el("episode").value = file.name.replace(/\.[^.]+$/, "");
  if (store.loadSaved(el("episode").value)) render();
  setStatus("Loaded " + file.name);
});

el("soundOn").addEventListener("change", e => video.muted = !e.target.checked);

/* ---- mask lock --------------------------------------------------------- */

el("lockMasks").addEventListener("change", () => {
  el("addMask").hidden = masksLocked();
  el("maskClass").hidden = masksLocked();
  render();
});

/* ---- hint popovers ----------------------------------------------------- */

/* Every ".hint" is a mark plus the body right after it; only one opens at a time. */
const hints = [...document.querySelectorAll(".hint")].map(hint => ({
  mark: hint.querySelector(".hint-mark"),
  body: hint.querySelector(".hint-body")
}));

function closeHints(except) {
  for (const hint of hints) {
    if (hint === except) continue;
    hint.body.hidden = true;
    hint.mark.setAttribute("aria-expanded", "false");
  }
}

for (const hint of hints) {
  hint.mark.addEventListener("click", () => {
    const open = hint.body.hidden;
    closeHints(hint);
    hint.body.hidden = !open;
    hint.mark.setAttribute("aria-expanded", String(open));
  });
}

document.addEventListener("click", e => {
  if (!e.target.closest(".hint")) closeHints();
});

/* ---- tags -------------------------------------------------------------- */

/* Suggestions come from the vocabulary in localStorage, so a tag typed once on
   any video is offered on the next one. */
function renderTagOptions() {
  el("tagOptions").innerHTML = tags.known
    .map(t => '<option value="' + escapeHtml(t) + '"></option>').join("");
}

function tagsHtml(list, locked) {
  const chips = list.map(t =>
    '<span class="chip">' + escapeHtml(t) +
    (locked ? "" : '<button type="button" class="chip-x" data-action="untag" data-tag="'
      + escapeHtml(t) + '" title="Remove tag">\u2715</button>') +
    '</span>').join("");
  const input = locked ? ""
    : '<input type="text" class="input tag-input" list="tagOptions" data-role="tagInput"'
      + ' placeholder="+ tag" autocomplete="off" spellcheck="false">';
  return '<div class="tags">' + chips + input + '</div>';
}

/* Tags waiting on the new-interval form, moved onto the row when it is added. */
let newTags = [];

function renderNewTags() {
  const list = newTags.map(t =>
    '<span class="chip">' + escapeHtml(t) +
    '<button type="button" class="chip-x" data-action="untag" data-tag="' + escapeHtml(t)
    + '" title="Remove tag">\u2715</button></span>').join("");
  el("newTags").innerHTML = list +
    '<input type="text" class="input tag-input" list="tagOptions" data-role="tagInput"'
    + ' placeholder="+ tag" autocomplete="off" spellcheck="false">';
}

el("newTags").addEventListener("click", e => {
  const button = e.target.closest('[data-action="untag"]');
  if (!button) return;
  const key = tags.clean(button.dataset.tag).toLowerCase();
  newTags = newTags.filter(t => t.toLowerCase() !== key);
  renderNewTags();
});

/* Takes what is in a tag input, files it in the vocabulary, hands it on. */
function takeTagInput(input, onTag) {
  const stored = tags.add(input.value);
  input.value = "";
  if (!stored) return;
  renderTagOptions();
  renderTagList();
  onTag(stored);
}

function pushNewTag(stored) {
  if (!newTags.some(t => t.toLowerCase() === stored.toLowerCase())) newTags.push(stored);
  renderNewTags();
}

el("newTags").addEventListener("keydown", e => {
  if (!e.target.matches('[data-role="tagInput"]')) return;
  if (e.key !== "Enter" && e.key !== ",") return;
  e.preventDefault();
  e.stopPropagation();
  takeTagInput(e.target, stored => {
    pushNewTag(stored);
    el("newTags").querySelector('[data-role="tagInput"]').focus();
  });
});

el("newTags").addEventListener("change", e => {
  if (!e.target.matches('[data-role="tagInput"]') || !e.target.value.trim()) return;
  takeTagInput(e.target, pushNewTag);
});

/* ---- tag panel --------------------------------------------------------- */

function renderTagList() {
  el("tagList").innerHTML = tags.known.map(t =>
    '<li class="tag-row" data-tag="' + escapeHtml(t) + '">' +
      '<span class="tag-name">' + escapeHtml(t) + '</span>' +
      '<span class="tag-count" title="rows carrying it">' + tagUsage(store.items, t) + '</span>' +
      '<button type="button" class="btn btn-icon btn-tiny" data-action="renameTag" title="Rename everywhere">\u270e</button>' +
      '<button type="button" class="btn btn-icon btn-tiny btn-danger" data-action="forgetTag" title="Remove from vocabulary and rows">\u2715</button>' +
    '</li>').join("");
  el("tagListEmpty").hidden = tags.known.length > 0;
}

el("tagManage").addEventListener("click", () => {
  const open = el("tagPanel").hidden;
  el("tagPanel").hidden = !open;
  el("tagManage").setAttribute("aria-expanded", String(open));
  if (open) { renderTagList(); el("tagNew").focus(); }
});

function closeTagPanel() {
  el("tagPanel").hidden = true;
  el("tagManage").setAttribute("aria-expanded", "false");
}
el("tagPanelClose").addEventListener("click", closeTagPanel);
document.addEventListener("click", e => {
  if (!e.target.closest(".tag-manage") && !el("tagPanel").hidden) closeTagPanel();
});

function addToVocabulary() {
  const stored = tags.add(el("tagNew").value);
  el("tagNew").value = "";
  if (!stored) return;
  renderTagOptions();
  renderTagList();
  setStatus('Tag "' + stored + '" is now remembered in this browser.');
}
el("tagAdd").addEventListener("click", addToVocabulary);
el("tagNew").addEventListener("keydown", e => {
  if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); addToVocabulary(); }
});

el("tagList").addEventListener("click", e => {
  const button = e.target.closest("button");
  const row = e.target.closest(".tag-row");
  if (!button || !row) return;
  const name = row.dataset.tag;
  const used = tagUsage(store.items, name);

  if (button.dataset.action === "renameTag") {
    const next = prompt('Rename "' + name + '" (' + used + " row(s) carry it):", name);
    if (next === null) return;
    const stored = tags.rename(name, next);
    if (!stored) { alert("That name is empty."); return; }
    for (const item of store.items) {
      const at = tagsOf(item).findIndex(t => t.toLowerCase() === name.toLowerCase());
      if (at < 0) continue;
      item.tags[at] = stored;
      item.tags = item.tags.filter((t, i) =>
        item.tags.findIndex(o => o.toLowerCase() === t.toLowerCase()) === i);
    }
    setStatus('Renamed "' + name + '" to "' + stored + '" on ' + used + " row(s).");
  } else if (button.dataset.action === "forgetTag") {
    if (used && !confirm('Remove "' + name + '" from the vocabulary and from '
      + used + " row(s)?")) return;
    tags.forget(name);
    for (const item of store.items) removeTagFrom(item, name);
    setStatus('Removed "' + name + '"' + (used ? " from " + used + " row(s)." : "."));
  } else return;

  renderTagOptions();
  renderTagList();
  render();
  autosave();
});

/* ---- new interval form ------------------------------------------------- */

/* How long an annotation may last, from the two header fields. Masks are never
   checked - the intro and the credits have whatever length they have.
   The defaults match the pipeline: 1.25 s is its frame sampling step, so a
   shorter annotation can fall between two sampled frames; above 15 s a
   described event spreads over several segments. */
function limits() {
  const min = +el("minDuration").value;
  const max = +el("maxDuration").value;
  return { min: min > 0 ? round2(min) : 0, max: max > 0 ? round2(max) : Infinity };
}

/* "" when the length is fine, otherwise "short" or "long". */
function durationProblem(start, end) {
  const { min, max } = limits();
  const duration = round2(end - start);
  if (duration < min) return "short";
  if (duration > max) return "long";
  return "";
}

function markDuration(element, problem) {
  element.classList.toggle("is-short", problem === "short");
  element.classList.toggle("is-long", problem === "long");
}

function refreshForm() {
  const start = el("newStart").value, end = el("newEnd").value;
  el("newStartCode").textContent = timeCode(start);
  el("newEndCode").textContent = timeCode(end);
  el("newWords").textContent = wordCount(el("newDescription").value) + " words";
  el("newWords").classList.toggle("is-short",
    wordCount(el("newDescription").value) < MIN_WORDS);
  const invalid = start !== "" && end !== "" && round2(+end) <= round2(+start);
  el("newStart").classList.toggle("is-invalid", invalid);
  el("newEnd").classList.toggle("is-invalid", invalid);

  const measurable = start !== "" && end !== "" && !isNaN(+start) && !isNaN(+end) && !invalid;
  el("newDuration").textContent = measurable ? round2(+end - +start).toFixed(2) + " s" : "";
  markDuration(el("newDuration"), measurable ? durationProblem(+start, +end) : "");
}
["newStart", "newEnd", "newDescription"].forEach(id => el(id).addEventListener("input", refreshForm));
/* Changing a limit re-judges the form and every row on the list. */
["minDuration", "maxDuration"].forEach(id => el(id).addEventListener("input", () => {
  refreshForm();
  refreshState();
}));

/* Reads the two time fields, or null when they don't make an interval. */
function formRange() {
  const startRaw = el("newStart").value, endRaw = el("newEnd").value;
  if (startRaw === "" || endRaw === "" || isNaN(+startRaw) || isNaN(+endRaw)) {
    alert("Enter start and end."); return null;
  }
  const start = round2(+startRaw), end = round2(+endRaw);
  if (end <= start) { alert("End must be later than start."); return null; }
  return { start, end };
}

const MIN_WORDS = 8;

function addItem(type) {
  const range = formRange();
  if (!range) return;

  const problem = type === "a" ? durationProblem(range.start, range.end) : "";
  if (problem) {
    const { min, max } = limits();
    alert("An annotation must last between " + min + " and " + max + " s; this one is "
      + round2(range.end - range.start).toFixed(2) + " s ("
      + (problem === "short" ? "too short" : "too long") + ").\n"
      + "Correct the times, or change the limits in the header.");
    el("newEnd").focus();
    return;
  }

  const description = el("newDescription").value.trim();
  if (type === "a" && wordCount(description) < MIN_WORDS) {
    alert("An annotation needs at least " + MIN_WORDS + " words of description.");
    el("newDescription").focus();
    return;
  }

  /* A mask carries its class as a tag: "przejscie" for a scene transition
     absorbed by the content axis, nothing for a structural one (logo, opening
     titles, closing credits) that cuts the episode into corpus ranges. The
     picker only saves typing - the tag is an ordinary one and can be edited
     on the row like any other. */
  const tagsForItem = newTags.slice();
  const maskClass = type === "m" ? el("maskClass").value : "";
  if (maskClass && !tagsForItem.includes(maskClass)) tagsForItem.push(maskClass);

  const item = {
    id: store.nextId(el("episode").value, type),
    type,
    start: range.start,
    end: range.end,
    description: description,
    tags: tagsForItem
  };

  const overlaps = store.maskOverlaps(item);
  if (overlaps.length) {
    alert("Masks cannot overlap each other: " + overlaps.join(", "));
    return;
  }

  const clashes = store.maskClashes(item);
  if (clashes.length) {
    alert(type === "m"
      ? "This mask would cover annotations: " + clashes.join(", ")
      : "An annotation cannot overlap a mask: " + clashes.join(", "));
    return;
  }

  store.add(item);
  tags.remember(item.tags);
  newTags = [];
  renderNewTags();
  renderTagList();
  el("newStart").value = el("newEnd").value = el("newDescription").value = "";
  refreshForm();
  render();
  autosave();
}

el("addAnnotation").addEventListener("click", () => addItem("a"));
el("addMask").addEventListener("click", () => addItem("m"));

el("endOfVideo").addEventListener("click", () => {
  if (!video.duration) { alert("Load a video first."); return; }
  el("newEnd").value = seconds(video.duration);
  refreshForm();
});

el("previewRange").addEventListener("click", () => {
  const range = formRange();
  if (range) player.playRange(range.start, range.end);
});

el("sortRows").addEventListener("click", () => { store.sortByStart(); render(); autosave(); });

/* ---- table ------------------------------------------------------------- */

const rowOf = id => el("rows").querySelector('tr.row-times[data-id="' + CSS.escape(id) + '"]');
const escapeHtml = s => s.replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const wordCount = text => (text.trim() ? text.trim().split(/\s+/).length : 0);
/* Last part of an id - the part that differs between rows and shows where the
   row came from: "d97650" imported with its source id, "001" created here,
   "m001" a mask. The full id sits in the cell's tooltip. */
const idTail = id => (id || "").split("_").pop();

/* Full rebuild - only on add / remove / import / sort / lock.
   An annotation is two rows: times + actions, then the description.
   A mask is a single row - its description is fixed. */
function render() {
  const body = el("rows");
  body.innerHTML = "";
  store.items.forEach((item, index) => {
    const mask = item.type === "m";
    const locked = mask && masksLocked();
    const disabled = locked ? " disabled" : "";

    const times = document.createElement("tr");
    times.dataset.id = item.id;
    times.className = "row-times" + (mask ? " row-mask" : "") +
      (locked && !item.description ? " row-single" : "");
    times.innerHTML = `
      <td class="num" title="${escapeHtml(item.id)}">
        ${index + 1}${mask ? '<span class="tag-mask">M</span>' : ""}
        <span class="row-id">${escapeHtml(idTail(item.id))}</span>
      </td>
      <td class="num">
        <input type="number" step="0.01" min="0" class="input input-num clock-input"
               value="${item.start.toFixed(2)}" data-field="start"${disabled}>
        <span class="mmss">${timeCode(item.start)}</span>
      </td>
      <td class="num">
        <input type="number" step="0.01" min="0" class="input input-num clock-input"
               value="${item.end.toFixed(2)}" data-field="end"${disabled}>
        <span class="mmss">${timeCode(item.end)}</span>
        ${locked ? "" : '<span class="duration row-duration"></span>'}
      </td>
      <td class="actions">
        <button class="btn btn-icon" data-action="play" title="Play interval">▶</button>
        ${locked ? "" : `
        <button class="btn btn-icon" data-action="setStart" title="Start = current time">[</button>
        <button class="btn btn-icon" data-action="setEnd" title="End = current time">]</button>
        <button class="btn btn-icon btn-danger" data-action="remove" title="Remove">✕</button>`}
      </td>`;
    body.appendChild(times);

    if (locked && !item.description) return;

    const description = document.createElement("tr");
    description.dataset.id = item.id;
    description.className = "row-description" + (mask ? " row-mask" : "");
    description.innerHTML = `
      <td></td>
      <td colspan="3">
        <div class="describe">
          <textarea class="input textarea textarea-sm" data-field="description"
            placeholder="${mask ? "Optional note, e.g. credits" : "Description"}"${disabled}>${escapeHtml(item.description)}</textarea>
          <span class="words${!mask && wordCount(item.description) < MIN_WORDS ? " is-short" : ""}">${wordCount(item.description)} words</span>
        </div>
        ${tagsHtml(tagsOf(item), locked)}
      </td>`;
    body.appendChild(description);
  });
  refreshState();
}

/* Light refresh: selection, overlap and error marks, durations, counters.
   No rebuild. */
function refreshState() {
  const { ids, pairs } = store.overlaps();
  const clashing = store.clashingIds();
  let outOfRange = 0;
  for (const row of el("rows").children) {
    const item = store.find(row.dataset.id);
    const problem = item && item.type === "a" ? durationProblem(item.start, item.end) : "";
    row.classList.toggle("is-active", row.dataset.id === store.activeId);
    row.classList.toggle("is-overlapping", ids.has(row.dataset.id));
    row.classList.toggle("is-out-of-range", !!problem);
    row.classList.toggle("is-invalid",
      clashing.has(row.dataset.id) || (!!item && item.end <= item.start));

    /* Only the times row carries a duration readout, so counting here counts
       every annotation exactly once. Masks get the readout too when unlocked,
       but never a short/long verdict - the limits describe annotations only. */
    const readout = row.querySelector(".row-duration");
    if (readout && item) {
      readout.textContent = round2(item.end - item.start).toFixed(2) + " s";
      markDuration(readout, problem);
      if (problem) outOfRange++;
    }
  }
  el("annotationCount").textContent = store.of("a").length;
  el("maskCount").textContent = store.of("m").length;
  el("overlapCount").textContent = pairs;
  const onMask = store.of("a").filter(i => store.maskClashes(i).length).length;
  el("onMaskCount").textContent = onMask;
  el("onMaskCount").classList.toggle("is-bad", onMask > 0);
  el("outOfRangeCount").textContent = outOfRange;
  el("outOfRangeCount").classList.toggle("is-bad", outOfRange > 0);
  el("emptyHint").hidden = store.items.length > 0;
}

/* ---- tags on a row ----------------------------------------------------- */

/* Repaints just the tag block of one row - a full render would blur the input. */
function refreshRowTags(item) {
  const row = el("rows").querySelector('tr.row-description[data-id="'
    + CSS.escape(item.id) + '"]');
  if (!row) return;
  const holder = row.querySelector(".tags");
  if (!holder) return;
  holder.outerHTML = tagsHtml(tagsOf(item), item.type === "m" && masksLocked());
  const input = row.querySelector('[data-role="tagInput"]');
  if (input) input.focus();
}

function rowTagFromInput(input) {
  const row = input.closest("tr");
  const item = row && store.find(row.dataset.id);
  if (!item) return;
  takeTagInput(input, stored => {
    addTagTo(item, stored);
    refreshRowTags(item);
    renderTagList();
    autosave();
  });
}

el("rows").addEventListener("keydown", e => {
  if (!e.target.matches('[data-role="tagInput"]')) return;
  if (e.key !== "Enter" && e.key !== ",") return;
  e.preventDefault();
  e.stopPropagation();
  rowTagFromInput(e.target);
});

el("rows").addEventListener("change", e => {
  if (!e.target.matches('[data-role="tagInput"]') || !e.target.value.trim()) return;
  rowTagFromInput(e.target);
});

/* Write to the model and to the visible field of that same row.

   Only one situation is refused here: a mask edited onto another mask. An
   annotation that ends up on a mask is marked and counted, but the edit goes
   through - a file may arrive with such a row (a TVR marker that starts inside
   a transition), and refusing the edit would make the row impossible to
   correct, which is the one thing the annotator exists for. Adding a NEW
   annotation on a mask is still refused, in addItem. */
function setField(item, field, value) {
  const previous = item[field];
  item[field] = value;
  const overlaps = store.maskOverlaps(item);
  if (overlaps.length) {
    item[field] = previous;
    alert("Masks cannot overlap each other: " + overlaps.join(", "));
  }
  const row = rowOf(item.id);
  if (!row) return;
  const input = row.querySelector('[data-field="' + field + '"]');
  input.value = item[field].toFixed(2);
  input.nextElementSibling.textContent = timeCode(item[field]);
}

el("rows").addEventListener("click", e => {
  const row = e.target.closest("tr");
  if (!row) return;
  const item = store.find(row.dataset.id);
  if (!item) return;
  store.activeId = item.id;

  const button = e.target.closest("button");
  if (!button) { refreshState(); return; }

  if (button.dataset.action === "untag") {
    removeTagFrom(item, button.dataset.tag);
    refreshRowTags(item);
    renderTagList();
    autosave();
    return;
  }

  switch (button.dataset.action) {
    case "play":
      player.playRange(item.start, item.end); refreshState(); break;
    case "setStart":
      setField(item, "start", round2(video.currentTime)); refreshState(); autosave(); break;
    case "setEnd":
      setField(item, "end", round2(video.currentTime)); refreshState(); autosave(); break;
    case "remove":
      if (!confirm('Remove "' + (item.description.slice(0, 40) || item.id) + '"?')) return;
      store.remove(item.id); render(); autosave(); break;
  }
});

el("rows").addEventListener("input", e => {
  const field = e.target.dataset.field;
  if (field === "start" || field === "end")
    e.target.nextElementSibling.textContent = timeCode(e.target.value);
  if (field === "description") {
    e.target.nextElementSibling.textContent = wordCount(e.target.value) + " words";
    const item = store.find(e.target.closest("tr").dataset.id);
    e.target.nextElementSibling.classList.toggle("is-short",
      !!item && item.type === "a" && wordCount(e.target.value) < MIN_WORDS);
  }
});

el("rows").addEventListener("change", e => {
  const row = e.target.closest("tr");
  const field = e.target.dataset.field;
  if (!row || !field) return;
  const item = store.find(row.dataset.id);
  if (!item) return;
  if (field === "description") item.description = e.target.value;
  else if (e.target.value === "" || isNaN(+e.target.value)) {
    e.target.value = item[field].toFixed(2);
    return;
  } else setField(item, field, round2(+e.target.value));
  refreshState();
  autosave();
});

/* The mouse wheel over a focused number input changes its value - while
   scrolling the list that would silently corrupt times. Drop focus instead. */
document.addEventListener("wheel", () => {
  const active = document.activeElement;
  if (active && active.tagName === "INPUT" && active.type === "number") active.blur();
}, { passive: true });

/* ---- keyboard ---------------------------------------------------------- */

document.addEventListener("keydown", e => {
  const inField = ["INPUT", "TEXTAREA"].includes(e.target.tagName);
  if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); addItem("a"); return; }
  if (inField && e.target.type !== "file") return;
  if (e.target.tagName === "BUTTON") e.target.blur();  // Space drives the video, not the button
  if (!video.src) return;

  const step = e.ctrlKey ? 1 / fps() : (e.shiftKey ? 0.1 : 1);
  switch (e.key) {
    case " ": e.preventDefault(); player.toggle(); break;
    case "ArrowLeft": e.preventDefault(); if (e.ctrlKey) video.pause(); player.step(-step); break;
    case "ArrowRight": e.preventDefault(); if (e.ctrlKey) video.pause(); player.step(step); break;
    case "s": case "S":
      el("newStart").value = seconds(video.currentTime); refreshForm(); break;
    case "e": case "E":
      el("newEnd").value = seconds(video.currentTime); refreshForm(); break;
  }
});

/* ---- CSV --------------------------------------------------------------- */

el("exportCsv").addEventListener("click", async () => {
  /* Overlapping masks are the one thing that cannot be waved through: the
     pipeline reads the masks as a partition of the episode, so a file with
     two of them over the same seconds is not merely suspicious, it is
     unusable. Everything else is a warning the annotator may accept. */
  const maskPairs = store.maskOverlapPairs();
  if (maskPairs.length) {
    alert("Masks overlap each other and the file cannot be saved:\n- "
      + maskPairs.join("\n- ") + "\n\nCorrect them and export again.");
    return;
  }

  const { duplicates, missing } = store.idProblems();
  const odd = store.of("a").filter(i => durationProblem(i.start, i.end));
  const onMask = store.of("a").filter(i => store.maskClashes(i).length);
  const { min, max } = limits();
  const trouble = [
    duplicates.length ? duplicates.length + " duplicated id(s): " + duplicates.join(", ") : "",
    missing ? missing + " row(s) without an id" : "",
    odd.length ? odd.length + " annotation(s) outside " + min + "-" + max + " s: "
      + odd.slice(0, 5).map(i => i.id).join(", ") : "",
    onMask.length ? onMask.length + " annotation(s) sitting on a mask: "
      + onMask.slice(0, 5).map(i => i.id).join(", ") : ""
  ].filter(Boolean);
  if (trouble.length && !confirm("Before saving:\n- " + trouble.join("\n- ") + "\n\nSave anyway?")) return;

  const name = (el("episode").value || "intervals").trim() + "_intervals.csv";
  /* Always exported in time order, whether or not "Sort by start" was ever
     clicked. The Python side writes the same order, so a file that travels
     through the annotator differs from the one that went in by the edits
     alone - never by the order of its rows. */
  const result = await saveCsv(name, toCsv(store.sorted()));
  if (!result.saved) { setStatus("Export cancelled"); return; }
  setStatus(result.picked
    ? "Saved as " + result.name
    : "Exported " + result.name + " to the downloads folder "
      + "(this browser does not let the page ask where to save)");
});

el("importCsv").addEventListener("click", () => el("csvFile").click());

el("csvFile").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;
  file.text().then(text => {
    const imported = fromCsv(text);
    if (!imported.length) { setStatus("No rows found in " + file.name); return; }
    /* Importing on top of a filled list is almost always a reload of the same
       file - appending would double every row. Replacing is the default. */
    if (store.items.length &&
        confirm("Replace the " + store.items.length + " row(s) on the list with " +
                imported.length + " from " + file.name + "?\nCancel = add them to the list.")) {
      store.items = [];
      store.activeId = null;
    }
    const { added, skipped } = store.merge(imported, el("episode").value);
    render();
    store.save(el("episode").value);
    const clashing = store.clashingIds().size;
    setStatus("Imported " + added + " of " + imported.length + " rows from " + file.name +
      (skipped.length ? " - skipped, id already on the list: " + skipped.join(", ") : "") +
      (clashing ? " - " + clashing + " rows overlap a mask" : ""));
  });
  e.target.value = "";
});

renderTagOptions();
renderTagList();
renderNewTags();
render();
