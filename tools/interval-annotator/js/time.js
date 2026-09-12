/* Time helpers. All stored times are seconds with 0.01 s precision. */

function round2(t) {
  return Math.round(t * 100) / 100;
}

/* Seconds -> "m:ss.cc" (display only). */
function timeCode(t) {
  if (t === "" || t === null || isNaN(t)) return "";
  const s = round2(Math.max(0, +t));
  const m = Math.floor(s / 60);
  return m + ":" + (s - m * 60).toFixed(2).padStart(5, "0");
}

function seconds(t) {
  return round2(t).toFixed(2);
}
