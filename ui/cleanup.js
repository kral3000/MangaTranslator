() => {
  if (window.__mangaCleanupSelection) return;
  window.__mangaCleanupSelection = true;
  let gesture = null;
  let synthetic = false;
  let suppressUntil = 0;
  let frame = 0;
  const anchors = new WeakMap();
  const rows = group => [...group.querySelectorAll('label')].filter(row => row.querySelector('input[type="checkbox"]'));
  const apply = (row, checked) => {
    const input = row.querySelector('input[type="checkbox"]');
    if (!input.disabled && input.checked !== checked) {
      synthetic = true;
      input.click();
      synthetic = false;
    }
  };
  const range = (group, first, last, checked, list = rows(group)) => {
    const a = list.indexOf(first), b = list.indexOf(last);
    if (a < 0 || b < 0) return apply(last, checked);
    list.slice(Math.min(a, b), Math.max(a, b) + 1).forEach(row => apply(row, checked));
  };
  const underPointer = () => {
    const viewport = gesture.viewport;
    const box = viewport.getBoundingClientRect();
    const top = Math.max(0, box.top), bottom = Math.min(window.innerHeight, box.bottom);
    const y = Math.max(top + 2, Math.min(bottom - 2, gesture.y));
    const row = document.elementFromPoint(gesture.x, y)?.closest('label');
    if (row && gesture.group.contains(row) && row.querySelector('input[type="checkbox"]')) {
      if (row !== gesture.last) {
        // Fill crossed rows even when a quick drag skips pointer events.
        range(gesture.group, gesture.last, row, gesture.checked, gesture.list);
        gesture.last = row;
      }
    }
  };
  const scroll = now => {
    if (!gesture) return;
    const viewport = gesture.viewport;
    const box = viewport.getBoundingClientRect();
    const top = Math.max(0, box.top), bottom = Math.min(window.innerHeight, box.bottom);
    const elapsed = Math.min(32, Math.max(0, now - gesture.time));
    gesture.time = now;
    let speed = 0;
    if (now >= gesture.wheelUntil && gesture.x >= box.left && gesture.x <= box.right && bottom > top) {
      const distance = gesture.y > bottom - 48 ? gesture.y - (bottom - 48)
        : gesture.y < top + 48 ? gesture.y - (top + 48) : 0;
      if (distance) speed = Math.sign(distance) * (60 + 660 * Math.min(1, Math.abs(distance) / 48) ** 2);
    }
    gesture.velocity += (speed - gesture.velocity) * (1 - Math.exp(-elapsed / 70));
    if (Math.abs(gesture.velocity) > 1) viewport.scrollTop += gesture.velocity * elapsed / 1000;
    underPointer();
    frame = requestAnimationFrame(scroll);
  };
  document.addEventListener('pointerdown', event => {
    if (event.button !== 0 || event.pointerType === 'touch') return;
    const group = event.target.closest('.cleanup-picker');
    const row = event.target.closest('label');
    const input = row?.querySelector('input[type="checkbox"]');
    if (!group || !input || input.disabled) return;
    const check = input.getBoundingClientRect();
    if (event.clientX < check.left - 12 || event.clientX > check.right + 12) return;
    event.preventDefault();
    input.focus({preventScroll: true});
    const checked = !input.checked;
    if (event.shiftKey && anchors.has(group)) range(group, anchors.get(group), row, checked);
    else apply(row, checked);
    anchors.set(group, row);
    const viewport = group.querySelector('[data-testid="checkbox-group"]') || group;
    gesture = {group, viewport, checked, last: row, list: rows(group), x: event.clientX, y: event.clientY,
      time: performance.now(), velocity: 0, wheelUntil: 0};
    group.setPointerCapture(event.pointerId);
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(scroll);
  }, true);
  document.addEventListener('pointermove', event => {
    if (!gesture) return;
    event.preventDefault();
    gesture.x = event.clientX; gesture.y = event.clientY;
    underPointer();
  }, {passive: false});
  const finish = () => {
    if (gesture) suppressUntil = performance.now() + 300;
    gesture = null;
    cancelAnimationFrame(frame);
  };
  document.addEventListener('pointerup', finish);
  document.addEventListener('pointercancel', finish);
  document.addEventListener('lostpointercapture', finish);
  window.addEventListener('blur', finish);
  document.addEventListener('wheel', event => {
    if (!gesture || !event.deltaY) return;
    event.preventDefault();
    const multiplier = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? gesture.viewport.clientHeight : 1;
    gesture.viewport.scrollTop += event.deltaY * multiplier;
    gesture.wheelUntil = performance.now() + 250;
    gesture.velocity = 0;
    underPointer();
  }, {capture: true, passive: false});
  document.addEventListener('pointerover', event => {
    const row = event.target.closest('.cleanup-picker label');
    if (row) row.title = row.textContent.trim();
  }, {passive: true});
  document.addEventListener('click', event => {
    if (synthetic) return;
    const group = event.target.closest('.cleanup-picker');
    if (!group) return;
    if (event.detail && (gesture || performance.now() < suppressUntil)) {
      event.preventDefault(); event.stopImmediatePropagation();
    } else {
      const row = event.target.closest('label');
      if (row) anchors.set(group, row);
    }
  }, true);
  document.addEventListener('keydown', event => {
    if (event.key !== ' ' || !event.shiftKey) return;
    const group = event.target.closest('.cleanup-picker');
    const row = event.target.closest('label');
    if (group && row && anchors.has(group)) {
      event.preventDefault();
      const input = row.querySelector('input[type="checkbox"]');
      range(group, anchors.get(group), row, !input.checked);
    }
  }, true);
}
