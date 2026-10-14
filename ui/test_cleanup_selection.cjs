// Exercise held-pointer scrolling without browser timing or filesystem effects.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const handlers = new Map(), frames = new Map();
let clock = 0, nextFrame = 0, offset = 0;
const viewport = {
  clientHeight: 320,
  get scrollTop() { return offset; },
  set scrollTop(value) { offset = Math.max(0, Math.min(960, value)); },
  getBoundingClientRect: () => ({left: 0, right: 800, top: 0, bottom: 320}),
};
const group = {
  querySelector(selector) {
    assert.equal(selector, '[data-testid="checkbox-group"]', 'use the item list, never the hidden loading overlay');
    return viewport;
  },
  querySelectorAll: () => list,
  contains: row => list.includes(row),
  setPointerCapture: () => {},
};
function dispatch(type, extra = {}) {
  const event = {target: list[0].input, preventDefault() {}, stopImmediatePropagation() {},
    button: 0, pointerType: 'mouse', pointerId: 1, clientX: 20, clientY: 16, ...extra};
  (handlers.get(type) || []).forEach(fn => fn(event));
}
const list = Array.from({length: 40}, (_, index) => {
  const row = {textContent: `job ${index}`, querySelector: () => row.input};
  row.closest = selector => selector === 'label' ? row : group;
  row.input = {
    checked: false, disabled: false, closest: row.closest, focus() {},
    getBoundingClientRect: () => ({left: 12, right: 28, top: index * 32 - offset}),
    click() { this.checked = !this.checked; dispatch('click', {target: this, detail: 0}); },
  };
  return row;
});
const document = {
  addEventListener(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); },
  elementFromPoint: (_x, y) => list[Math.floor((y + offset) / 32)],
};
const window = {innerHeight: 900, addEventListener: document.addEventListener};
vm.runInNewContext(`(${fs.readFileSync(path.join(__dirname, 'cleanup.js'), 'utf8')})()`, {
  window, document, performance: {now: () => clock},
  requestAnimationFrame(fn) { const id = ++nextFrame; frames.set(id, fn); return id; },
  cancelAnimationFrame(id) { frames.delete(id); },
});
function step(ms = 16) {
  clock += ms;
  const pending = [...frames.values()]; frames.clear(); pending.forEach(fn => fn(clock));
}
function down(index) {
  dispatch('pointerdown', {target: list[index].input, clientY: index * 32 - offset + 16});
}
function move(index) { dispatch('pointermove', {clientY: index * 32 - offset + 16}); }
const selected = () => list.flatMap((row, index) => row.input.checked ? [index + 1] : []);
down(0); move(3); dispatch('pointerup');
viewport.scrollTop = 9 * 32;
down(9); move(13); dispatch('pointerup');
assert.deepEqual(selected(), [1, 2, 3, 4, 10, 11, 12, 13, 14]);
down(9);
dispatch('wheel', {deltaY: 128, deltaMode: 0});
assert.deepEqual(selected(), [1, 2, 3, 4]);
const afterWheel = offset; step(160);
assert.equal(offset, afterWheel, 'automatic scroll must yield to the wheel');
dispatch('pointerup');
list.forEach(row => { row.input.checked = false; }); viewport.scrollTop = 0;
down(0); dispatch('pointermove', {clientY: 317});
for (let n = 0; n < 30; n++) step();
assert(offset > 100, 'edge scrolling must continue while the pointer is held');
assert.deepEqual(selected(), Array.from({length: selected().length}, (_, i) => i + 1));
dispatch('pointerup'); const stopped = offset; step(100);
assert.equal(offset, stopped, 'releasing must stop edge scrolling');
console.log('PASS: disjoint drag, held-pointer wheel, continuous edge selection, release');
