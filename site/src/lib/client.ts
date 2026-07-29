const $ = (s: string, r: ParentNode = document) => r.querySelector<HTMLElement>(s);
const $$ = (s: string, r: ParentNode = document) => Array.from(r.querySelectorAll<HTMLElement>(s));

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function initThemeToggle() {
  $("#theme-toggle")?.addEventListener("click", () => {
    const dark = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("lb-theme", dark ? "dark" : "light");
    } catch {
      /* storage unavailable */
    }
    window.dispatchEvent(new CustomEvent("lb-theme", { detail: dark }));
  });
}

function countUp(el: HTMLElement) {
  const target = parseFloat(el.dataset.countTo || "0");
  const isInt = el.dataset.format === "int";
  const render = (v: number) => {
    el.textContent = isInt ? Math.round(v).toLocaleString("en-US") : v.toFixed(1);
  };
  if (reduceMotion) {
    render(target);
    return;
  }
  const dur = 950;
  const start = performance.now();
  const ease = (x: number) => 1 - Math.pow(1 - x, 3);
  const step = (now: number) => {
    const p = Math.min(1, (now - start) / dur);
    render(target * ease(p));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initReveals() {
  const els = $$(".reveal");
  if (reduceMotion) {
    els.forEach((el) => {
      el.classList.add("in");
      $$(".bar > i", el).forEach((b) => {
        if (b.dataset.w) b.style.width = b.dataset.w;
      });
    });
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const el = e.target as HTMLElement;
        el.classList.add("in");
        $$(".bar > i", el).forEach((b) => {
          if (b.dataset.w) b.style.width = b.dataset.w;
        });
        $$("[data-count-to]", el).forEach(countUp);
        io.unobserve(el);
      }
    },
    { threshold: 0.12 },
  );
  els.forEach((el) => io.observe(el));
}

function initLeaderboard() {
  const tbody = $("#lb-body");
  if (!tbody) return;

  const rows = () => $$(".lb-row", tbody);
  const detailFor = (row: HTMLElement) =>
    row.nextElementSibling?.classList.contains("lb-detail") ? row.nextElementSibling : null;

  function sortRows(key: string, dir: 1 | -1) {
    const attr = `data-${key}`;
    const pairs = rows().map((row) => {
      const raw = row.getAttribute(attr);
      const v = raw == null || raw === "" ? null : parseFloat(raw);
      return { row, detail: detailFor(row), v };
    });
    pairs.sort((a, b) => {
      if (a.v == null && b.v == null) return 0;
      if (a.v == null) return 1;
      if (b.v == null) return -1;
      return (b.v - a.v) * dir;
    });
    pairs.forEach((p, i) => {
      tbody.appendChild(p.row);
      if (p.detail) tbody.appendChild(p.detail);
      const rank = $(".lb-rank", p.row);
      if (rank) rank.textContent = String(i + 1);
    });
  }

  $$(".lb-tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      $$(".lb-tab").forEach((x) => x.setAttribute("data-active", "false"));
      tab.setAttribute("data-active", "true");
      sortRows(tab.dataset.tab ?? "overall", 1);
    }),
  );

  const dirs: Record<string, 1 | -1> = {};
  $$(".lb-sort").forEach((th) =>
    th.addEventListener("click", () => {
      const key = th.dataset.sort ?? "overall";
      dirs[key] = dirs[key] === 1 ? -1 : 1;
      $$(".lb-sort").forEach((o) => o.removeAttribute("data-dir"));
      th.setAttribute("data-dir", dirs[key] === 1 ? "desc" : "asc");
      sortRows(key, dirs[key]);
    }),
  );

  rows().forEach((row) =>
    row.addEventListener("click", () => {
      const d = detailFor(row);
      if (!d) return;
      const open = d.classList.toggle("lb-open");
      row.classList.toggle("open", open);
    }),
  );
}

initThemeToggle();
initReveals();
initLeaderboard();

if (document.querySelector("[data-chart]")) {
  import("./charts").then(({ initCharts }) => initCharts());
}
