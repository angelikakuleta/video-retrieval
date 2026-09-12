/* CSV in/out. Format: id;type;start;end;desc;tags
   type is "event" (annotation) or "mask". Dot as decimal separator.
   tags is a comma separated list of free-form labels (empty when there are none). */

const CSV_TYPE = { a: "event", m: "mask" };

function toCsv(items) {
  const escape = v => (/[;"\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v);
  const head = "id;type;start;end;desc;tags";
  const body = items.map(i =>
    [i.id, CSV_TYPE[i.type], i.start.toFixed(2), i.end.toFixed(2), escape(i.description),
     escape((Array.isArray(i.tags) ? i.tags : []).join(","))].join(";"));
  /* CRLF: every CSV in the pipeline uses it, and so does Excel. */
  return [head, ...body].join("\r\n") + "\r\n";
}

/* Minimal RFC 4180 parser with ';' as separator. Returns rows of strings. */
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c !== '"') field += c;
      else if (text[i + 1] === '"') { field += '"'; i++; }
      else quoted = false;
    } else if (c === '"') quoted = true;
    else if (c === ";") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.some(v => v !== ""));
}

/* CSV text -> items. Older files without a type column import as annotations. */
function fromCsv(text) {
  const rows = parseCsv(text.replace(/^\uFEFF/, ""));
  if (!rows.length) return [];
  const head = rows.shift().map(h => h.trim().toLowerCase());
  const at = (row, ...names) => {
    for (const n of names) {
      const k = head.indexOf(n);
      if (k >= 0) return (row[k] || "").trim();
    }
    return "";
  };
  const num = v => round2(+String(v).replace(",", ".") || 0);
  return rows.filter(r => r.length >= 3).map(r => {
    const raw = at(r, "type").toLowerCase();
    return {
      id: at(r, "id", "event_id"),
      type: raw === "mask" || raw === "m" ? "m" : "a",
      start: num(at(r, "start")),
      end: num(at(r, "end", "koniec")),
      description: at(r, "desc", "description", "opis"),
      tags: at(r, "tags")
        .split(",").map(t => t.trim().replace(/\s+/g, "_")).filter(Boolean)
    };
  });
}

function downloadCsv(fileName, text) {
  const blob = new Blob(["\uFEFF" + text], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(link.href);
}

/* The file chosen last time, so the next export opens the picker in the same
   folder - with one file per episode that is the whole difference between
   "Save as" being useful and being a chore. */
let lastSaveHandle = null;

/* Asks where to save, and writes the file there.

   Uses the File System Access API, which lets the file land straight in
   `data/annotations/<dataset>/` instead of the downloads folder. Browsers that
   do not have it (Firefox, Safari), and pages the browser refuses to give it to
   (a `file://` page in Chrome), fall back to a plain download - the export still
   works, it just goes where downloads go.

   Returns {saved, name, picked}: `picked` says whether the user chose the
   location, `saved` is false only when the picker was cancelled. */
async function saveCsv(fileName, text) {
  if (!window.showSaveFilePicker) {
    downloadCsv(fileName, text);
    return { saved: true, name: fileName, picked: false };
  }
  const options = {
    suggestedName: fileName,
    types: [{ description: "CSV", accept: { "text/csv": [".csv"] } }]
  };
  if (lastSaveHandle) options.startIn = lastSaveHandle;
  let handle;
  try {
    handle = await window.showSaveFilePicker(options);
  } catch (error) {
    if (error && error.name === "AbortError") {
      return { saved: false, name: fileName, picked: true };
    }
    downloadCsv(fileName, text);   // the browser would not open the picker
    return { saved: true, name: fileName, picked: false };
  }
  const stream = await handle.createWritable();
  await stream.write(new Blob(["\uFEFF" + text], { type: "text/csv;charset=utf-8" }));
  await stream.close();
  lastSaveHandle = handle;
  return { saved: true, name: handle.name, picked: true };
}
