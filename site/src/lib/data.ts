import raw from "../data/summary.json";

export interface BenchScore {
  score: number | null;
  correct: number;
  total: number;
  wilson: number;
}

export interface ModelSummary {
  name: string;
  model_id: string;
  provider: string;
  thinking_effort: string | null;
  overall: number | null;
  categories: Record<string, number | null>;
  benchmarks: Record<string, BenchScore>;
  completed_benchmarks: number;
  tokens: { input: number; output: number; thinking: number; total: number };
  cost_usd: number | null;
  avg_tps: number | null;
  avg_time_ms: number | null;
}

export interface CategoryMeta {
  key: string;
  label: string;
  icon: string;
  benchmarks: string[];
}

export interface BenchmarkMeta {
  key: string;
  display: string;
  category: string;
  full_dataset: string;
  sampled: number;
  verification: string;
  source: string;
  paper: string;
  description: string;
}

export interface Summary {
  generated_at: string;
  run_timestamp: string | null;
  seed: number;
  temperature: number;
  max_tokens: number;
  categories: CategoryMeta[];
  benchmarks: BenchmarkMeta[];
  models: ModelSummary[];
}

export const summary = raw as Summary;

export const MODEL_COLORS = [
  "#f0b429",
  "#2dd4bf",
  "#60a5fa",
  "#f472b6",
  "#a78bfa",
  "#34d399",
  "#fb923c",
  "#f87171",
  "#38bdf8",
  "#e879f9",
];

export const modelColor = (i: number): string => MODEL_COLORS[i % MODEL_COLORS.length];

export const CATEGORY_HUES: Record<string, string> = {
  coding: "#f0b429",
  science: "#2dd4bf",
  math: "#60a5fa",
  knowledge: "#a78bfa",
  instruction: "#f472b6",
};

export const pct = (v: number | null, digits = 1): string =>
  v == null ? "—" : `${(v * 100).toFixed(digits)}%`;

export const fmtInt = (v: number): string => v.toLocaleString("en-US");

export const fmtCost = (v: number | null): string =>
  v == null ? "—" : `$${v.toFixed(v < 1 ? 4 : 2)}`;

export const fmtTps = (v: number | null): string => (v == null ? "—" : v.toFixed(1));

export const fmtMs = (v: number | null): string =>
  v == null ? "—" : `${(v / 1000).toFixed(1)}s`;

export const fmtDate = (iso: string | null): string => {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
};

/** Red → amber → green heat ramp usable as text color in both themes. */
export const heatColor = (v: number | null): string =>
  v == null ? "var(--muted)" : `hsl(${Math.round(v * 140)} 68% 46%)`;

export const hfUrl = (source: string): string | null => {
  const slug = source.split(" ")[0];
  return slug.includes("/") ? `https://huggingface.co/datasets/${slug}` : null;
};
