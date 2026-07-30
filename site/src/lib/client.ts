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
  if (el.dataset.counted === "true") return;
  el.dataset.counted = "true";
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

function triggerReveal(el: HTMLElement) {
  el.classList.add("in");
  $$(".bar > i", el).forEach((b) => {
    if (b.dataset.w) b.style.width = b.dataset.w;
  });
  $$("[data-count-to]", el).forEach(countUp);
}

function initReveals() {
  const els = $$(".reveal");
  if (reduceMotion) {
    els.forEach(triggerReveal);
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        triggerReveal(e.target as HTMLElement);
        io.unobserve(e.target);
      }
    },
    { threshold: 0.05 },
  );
  els.forEach((el) => {
    io.observe(el);
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      triggerReveal(el);
      io.unobserve(el);
    }
  });
  $$("[data-count-to]").forEach((el) => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      countUp(el);
    }
  });
}

function initLeaderboard() {
  const tbody = $("#lb-body");
  if (!tbody) return;

  const rows = () => $$(".lb-row", tbody);
  const detailFor = (row: HTMLElement) =>
    row.nextElementSibling?.classList.contains("lb-detail") ? row.nextElementSibling : null;

  const searchInput = $("#lb-search") as HTMLInputElement | null;
  const providerSelect = $("#lb-provider-select") as HTMLSelectElement | null;
  const countIndicator = $("#lb-count-indicator");

  function filterRows() {
    const query = searchInput?.value.trim().toLowerCase() || "";
    const selectedProvider = providerSelect?.value.toLowerCase() || "";
    let visibleCount = 0;
    const totalCount = rows().length;

    rows().forEach((row) => {
      const name = row.dataset.name || "";
      const provider = row.dataset.provider || "";
      const modelId = row.dataset.modelId || "";
      const detail = detailFor(row);

      const matchesSearch = !query || name.includes(query) || provider.includes(query) || modelId.includes(query);
      const matchesProvider = !selectedProvider || provider === selectedProvider;

      if (matchesSearch && matchesProvider) {
        row.style.display = "";
        visibleCount++;
      } else {
        row.style.display = "none";
        if (detail) detail.style.display = "none";
        row.classList.remove("open");
      }
    });

    if (countIndicator) {
      countIndicator.textContent = `Showing ${visibleCount} of ${totalCount} models`;
    }
  }

  searchInput?.addEventListener("input", filterRows);
  providerSelect?.addEventListener("change", filterRows);

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
      if (rank) {
        if (key === "overall" && dir === 1) {
          if (i === 0) rank.textContent = "🥇";
          else if (i === 1) rank.textContent = "🥈";
          else if (i === 2) rank.textContent = "🥉";
          else rank.textContent = String(i + 1);
        } else {
          rank.textContent = String(i + 1);
        }
      }
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
      if (open) d.style.display = "table-row";
      else d.style.display = "none";
    }),
  );
}

function initNavigation() {
  const sections = $$("section[id]");
  const navLinks = $$(".nav-link");

  const io = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute("id");
          navLinks.forEach((link) => {
            const href = link.getAttribute("href");
            if (href === `#${id}`) {
              link.classList.add("active");
            } else {
              link.classList.remove("active");
            }
          });
        }
      }
    },
    { threshold: 0.25 },
  );

  sections.forEach((sec) => io.observe(sec));

  // Back to Top Button
  const btt = $("#back-to-top");
  if (btt) {
    window.addEventListener("scroll", () => {
      if (window.scrollY > 400) {
        btt.classList.remove("opacity-0", "pointer-events-none");
        btt.classList.add("opacity-100", "pointer-events-auto");
      } else {
        btt.classList.add("opacity-0", "pointer-events-none");
        btt.classList.remove("opacity-100", "pointer-events-auto");
      }
    });

    btt.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  // Mobile Menu Toggle
  const mobileBtn = $("#mobile-menu-toggle");
  const mobileMenu = $("#mobile-menu");
  if (mobileBtn && mobileMenu) {
    mobileBtn.addEventListener("click", () => {
      mobileMenu.classList.toggle("hidden");
    });
    $$(".mobile-nav-link").forEach((link) => {
      link.addEventListener("click", () => {
        mobileMenu.classList.add("hidden");
      });
    });
  }
}

function initChartExpansion() {
  $$("[data-expand-chart]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const chartId = btn.dataset.expandChart;
      const chartEl = $(`[data-chart="${chartId}"]`);
      if (!chartEl) return;

      const backdrop = document.createElement("div");
      backdrop.className = "chart-modal-backdrop";
      
      const modal = document.createElement("div");
      modal.className = "relative w-full max-w-5xl rounded-2xl border border-edge bg-panel p-6 shadow-2xl";
      
      modal.innerHTML = `
        <div class="flex items-center justify-between pb-4 border-b border-edge">
          <h3 class="font-display text-lg font-semibold tracking-tight">Expanded View</h3>
          <button type="button" class="close-modal text-muted hover:text-fg font-mono text-xl p-1">✕</button>
        </div>
        <div class="modal-chart-container h-[70vh] w-full mt-4"></div>
      `;

      backdrop.appendChild(modal);
      document.body.appendChild(backdrop);

      const modalContainer = modal.querySelector(".modal-chart-container") as HTMLElement;

      import("./charts").then(({ initCharts }) => {
        chartEl.setAttribute("data-expanded", "true");
        const clonedEl = document.createElement("div");
        clonedEl.className = "w-full h-full";
        clonedEl.setAttribute("data-chart", chartId);
        modalContainer.appendChild(clonedEl);
        initCharts();
      });

      const closeModal = () => {
        backdrop.remove();
      };

      modal.querySelector(".close-modal")?.addEventListener("click", closeModal);
      backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) closeModal();
      });
    });
  });
}

initThemeToggle();
initReveals();
initLeaderboard();
initNavigation();
initChartExpansion();

if (document.querySelector("[data-chart]")) {
  import("./charts").then(({ initCharts }) => initCharts());
}

