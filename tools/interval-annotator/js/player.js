/* Video playback: stepping and looping a single interval. */

const player = {
  video: null,
  getFps: () => 25,
  loop: null,

  init(video, getFps) {
    this.video = video;
    this.getFps = getFps;
    // A focused <video controls> handles Space/arrows itself, which would
    // double-toggle our shortcuts - so hand focus straight back to the document.
    video.addEventListener("focus", () => video.blur());
  },

  step(delta) {
    this.stopLoop();
    const max = this.video.duration || 1e9;
    this.video.currentTime = Math.max(0, Math.min(max, this.video.currentTime + delta));
  },

  toggle() {
    this.stopLoop();
    this.video.paused ? this.video.play() : this.video.pause();
  },

  /* Play start..end and stop exactly at the end. Checked once per animation
     frame - 'timeupdate' fires ~4x/s, which would overshoot noticeably. */
  playRange(start, end) {
    this.stopLoop();
    this.video.currentTime = start;
    this.video.play();
    const tick = () => {
      if (this.loop !== tick) return;
      if (this.video.currentTime >= end) {
        this.video.pause();
        this.video.currentTime = end;
        this.loop = null;
        return;
      }
      requestAnimationFrame(tick);
    };
    this.loop = tick;
    requestAnimationFrame(tick);
  },

  stopLoop() {
    this.loop = null;
  }
};
