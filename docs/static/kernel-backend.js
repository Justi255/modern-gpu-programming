/* Extend sphinx-design's native per-session sync with a durable book choice.
 * All examples still render as ordinary accessible radio tabs without this JS.
 */
(() => {
  "use strict";
  const group = "kernel-backend";
  const key = "fa5060-kernel-backend";
  const valid = (value) => value === "tilelang" || value === "triton";
  const labels = () => Array.from(document.querySelectorAll(
    `.sd-tab-label[data-sync-group="${group}"]`
  ));
  function select(backend, persist) {
    if (!valid(backend)) return;
    for (const label of labels()) {
      const input = label.previousElementSibling;
      if (input instanceof HTMLInputElement) {
        input.checked = label.dataset.syncId === backend;
      }
    }
    try {
      // The extension owns synchronization; mirror its key for native handlers.
      window.sessionStorage.setItem(`sphinx-design-tab-id-${group}`, backend);
      if (persist) window.localStorage.setItem(key, backend);
    } catch (_) {
      // Disabled storage must not disable tab interaction.
    }
  }
  function ready() {
    let saved;
    try { saved = window.localStorage.getItem(key); } catch (_) {}
    const requested = new URLSearchParams(window.location.search).get(group);
    select(valid(requested) ? requested : valid(saved) ? saved : "tilelang", true);
    for (const label of labels()) {
      label.addEventListener("click", () => select(label.dataset.syncId, true));
      const input = label.previousElementSibling;
      if (input instanceof HTMLInputElement) {
        input.addEventListener("change", () => {
          if (input.checked) select(label.dataset.syncId, true);
        });
      }
    }
    function revealAnchor() {
      let id;
      try { id = decodeURIComponent(window.location.hash.slice(1)); } catch (_) { return; }
      const target = id && document.getElementById(id);
      const panel = target && target.closest(".sd-tab-content");
      const label = panel && panel.previousElementSibling;
      if (label instanceof HTMLElement && valid(label.dataset.syncId)) {
        // Existing links into the dated Triton report must remain readable.
        select(label.dataset.syncId, false);
        window.requestAnimationFrame(() => target.scrollIntoView());
      }
    }
    revealAnchor();
    window.addEventListener("hashchange", revealAnchor);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", ready);
  } else {
    ready();
  }
})();
