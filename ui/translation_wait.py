"""The result-area wait display, driven by completed translation outputs."""

from threading import Lock

from .cancellation import CancellationManager


class TranslationWait:
    def __init__(self, *, batch=False):
        self.batch = batch
        self.cancellation = CancellationManager()
        self._lock = Lock()
        self._progress = (0, None if batch else 1, "running")
        self._delivered = None

    def report(self, completed, total):
        with self._lock:
            if self._progress[2] == "running":
                self._progress = (completed, total, "running")

    def finish(self, state):
        with self._lock:
            completed, total, previous = self._progress
            if previous == "running":
                self._progress = (completed, total, state)

    def render(self, *, only_changed=False):
        with self._lock:
            snapshot = self._progress
            if only_changed and snapshot == self._delivered:
                return None
            self._delivered = snapshot
        completed, total, state = snapshot
        label = {
            "running": "번역 중…",
            "complete": "번역 완료",
            "partial": "일부 완료",
            "cancelled": "번역 중단",
            "failed": "번역 실패",
        }[state]
        count = f"{completed} / {total}장 완료" if total is not None else (
            "전체 장수 확인 중" if state == "running" else f"{completed}장 완료"
        )
        meter = (
            f'<progress value="{completed}" max="{total}" aria-label="번역 완료 장수"></progress>'
            if total else ""
        )
        return (
            '<div class="translation-readout" role="status" aria-atomic="true">'
            f'<div><strong>{label}</strong><span>{count}</span></div>{meter}</div>'
        )


MOTION_HTML = """
<div class="translation-motion" aria-hidden="true">
  <div class="comic-scene is-current">
    <div class="wait-balloon large-balloon"><span lang="ja">またね</span></div>
    <div class="wait-ocr"><i></i></div>
  </div>
  <div class="comic-scene">
    <div class="wait-balloon source-balloon"><span lang="ja">またね</span></div>
    <i class="wait-arrow"></i>
    <div class="wait-balloon translated-balloon"><span>또 봐</span></div>
  </div>
  <div class="comic-scene">
    <div class="wait-balloon large-balloon">
      <span class="wait-erase" lang="ja">またね</span><i class="wait-mask"></i>
    </div>
  </div>
  <div class="comic-scene">
    <div class="wait-balloon large-balloon"><span class="wait-letter">또 봐</span><i class="wait-bounds"></i></div>
  </div>
  <div class="comic-scene">
    <div class="wait-background">
      <div class="wait-covered wait-erase"><span lang="ja">ザッ</span></div>
      <i class="wait-mask"></i>
    </div>
  </div>
</div>
"""

# Progress is rendered separately so a completed page doesn't restart the animation.
MOTION_HEAD = """
<script>
(() => {
    const orders = new WeakMap();
    document.addEventListener("animationend", (event) => {
        const current = event.target;
        if (event.animationName !== "comic-wait-cycle") return;
        const motion = current.parentElement;
        if (!motion?.classList.contains("translation-motion") || !motion.getClientRects().length) return;
        let order = orders.get(motion);
        if (!order?.length) {
            order = [...motion.children];
            for (let index = order.length - 1; index > 0; index--) {
                const other = Math.floor(Math.random() * (index + 1));
                [order[index], order[other]] = [order[other], order[index]];
            }
            if (order[order.length - 1] === current) order.unshift(order.pop());
            orders.set(motion, order);
        }
        current.classList.remove("is-current");
        order.pop().classList.add("is-current");
    });
})();
</script>
"""
