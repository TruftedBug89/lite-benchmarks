import { BarChart, HeatmapChart, RadarChart, ScatterChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  RadarComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { MODEL_COLORS, modelColor, summary } from "./data";

echarts.use([
  BarChart,
  HeatmapChart,
  RadarChart,
  ScatterChart,
  GridComponent,
  LegendComponent,
  RadarComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
]);

type Opt = echarts.EChartsOption;

const MONO = '"JetBrains Mono", ui-monospace, monospace';

interface Theme {
  fg: string;
  muted: string;
  grid: string;
  axis: string;
  panel: string;
  edge: string;
  base: string;
}

const theme = (dark: boolean): Theme =>
  dark
    ? {
        fg: "#e7ebf5",
        muted: "#8d96ad",
        grid: "rgba(146,162,200,0.08)",
        axis: "#2a3350",
        panel: "#1a2234",
        edge: "#2a3350",
        base: "#0e131f",
      }
    : {
        fg: "#171f31",
        muted: "#5c6577",
        grid: "rgba(18,25,45,0.07)",
        axis: "#d8dbe2",
        panel: "#ffffff",
        edge: "#e2e4ea",
        base: "#f6f6f2",
      };

const tooltip = (t: Theme) => ({
  backgroundColor: t.panel,
  borderColor: t.edge,
  borderWidth: 1,
  padding: [8, 12] as [number, number],
  textStyle: { color: t.fg, fontSize: 12, fontFamily: MONO },
  extraCssText: "border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.28);",
});

const legend = (t: Theme) => ({
  bottom: 0,
  icon: "circle",
  itemWidth: 8,
  itemHeight: 8,
  itemGap: 16,
  textStyle: { color: t.muted, fontSize: 11, fontFamily: MONO },
});

const axisLabel = (t: Theme) => ({ color: t.muted, fontFamily: MONO, fontSize: 11 });

const scored = summary.models.filter((m) => m.overall != null);

const kFmt = (v: number): string =>
  v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `${Math.round(v / 1e3)}K` : `${v}`;

function costOption(t: Theme): Opt {
  const pts = scored.filter((m) => m.cost_usd != null);
  const maxTps = Math.max(...pts.map((m) => m.avg_tps ?? 0), 1);
  return {
    grid: { left: 52, right: 28, top: 24, bottom: 58 },
    tooltip: {
      ...tooltip(t),
      trigger: "item",
      formatter: (p: any) => {
        const [cost, score, tps, name] = p.data;
        return `<b style="color:${t.fg}">${name}</b><br/>score ${score.toFixed(1)} · $${cost.toFixed(4)}/run<br/>${tps} tok/s out`;
      },
    },
    xAxis: {
      type: "value",
      scale: true,
      name: "est. cost per full run (USD)",
      nameLocation: "middle",
      nameGap: 34,
      nameTextStyle: { color: t.muted, fontFamily: MONO, fontSize: 11 },
      axisLabel: { ...axisLabel(t), formatter: "${value}" },
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { lineStyle: { color: t.axis } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 100,
      axisLabel: axisLabel(t),
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { show: false },
    },
    series: [
      {
        type: "scatter",
        symbolSize: (d: number[]) => 12 + ((d[2] as number) / maxTps) * 26,
        data: pts.map((m, i) => ({
          value: [m.cost_usd, (m.overall ?? 0) * 100, m.avg_tps ?? 0, m.name],
          itemStyle: { color: modelColor(summary.models.indexOf(m)), opacity: 0.88 },
        })),
        emphasis: { scale: 1.25 },
      },
    ],
  };
}

function radarOption(t: Theme): Opt {
  const top = scored.slice(0, 5);
  const cats = summary.categories.filter((c) => top.some((m) => m.categories[c.key] != null));
  return {
    tooltip: { ...tooltip(t), trigger: "item" },
    legend: { ...legend(t), data: top.map((m) => m.name) },
    radar: {
      indicator: cats.map((c) => ({ name: `${c.icon} ${c.label}`, max: 100 })),
      center: ["50%", "46%"],
      radius: "64%",
      axisName: { color: t.muted, fontFamily: MONO, fontSize: 11 },
      splitLine: { lineStyle: { color: t.grid } },
      splitArea: { show: false },
      axisLine: { lineStyle: { color: t.grid } },
    },
    series: [
      {
        type: "radar",
        data: top.map((m) => ({
          name: m.name,
          value: cats.map((c) => Math.round((m.categories[c.key] ?? 0) * 1000) / 10),
          lineStyle: { width: 2, color: modelColor(summary.models.indexOf(m)) },
          itemStyle: { color: modelColor(summary.models.indexOf(m)) },
          areaStyle: { color: modelColor(summary.models.indexOf(m)), opacity: 0.07 },
          symbol: "circle",
          symbolSize: 4,
        })),
      },
    ],
  };
}

function catbarsOption(t: Theme): Opt {
  const top = scored.slice(0, 5);
  const cats = summary.categories;
  return {
    grid: { left: 40, right: 12, top: 16, bottom: 64 },
    tooltip: {
      ...tooltip(t),
      trigger: "axis",
      axisPointer: { type: "shadow", shadowStyle: { color: t.grid } },
      valueFormatter: (v: any) => (v == null ? "—" : `${Number(v).toFixed(1)}%`),
    },
    legend: { ...legend(t), data: top.map((m) => m.name) },
    xAxis: {
      type: "category",
      data: cats.map((c) => `${c.icon} ${c.label}`),
      axisLabel: { ...axisLabel(t), interval: 0 },
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      max: 100,
      axisLabel: { ...axisLabel(t), formatter: "{value}%" },
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { show: false },
    },
    series: top.map((m) => ({
      name: m.name,
      type: "bar" as const,
      barGap: "12%",
      barMaxWidth: 13,
      itemStyle: {
        color: modelColor(summary.models.indexOf(m)),
        borderRadius: [3, 3, 0, 0],
      },
      data: cats.map((c) =>
        m.categories[c.key] == null ? null : Math.round((m.categories[c.key] as number) * 1000) / 10,
      ),
    })),
  };
}

function heatmapOption(t: Theme): Opt {
  const models = summary.models;
  const benches = summary.benchmarks;
  const data: [number, number, number | null, number, number, number][] = [];
  models.forEach((m, y) =>
    benches.forEach((b, x) => {
      const e = m.benchmarks[b.key];
      data.push([x, y, e && e.score != null ? Math.round(e.score * 100) : null, e?.correct ?? 0, e?.total ?? 0, e?.wilson ?? 0]);
    }),
  );
  return {
    grid: { left: 138, right: 16, top: 8, bottom: 96 },
    tooltip: {
      ...tooltip(t),
      formatter: (p: any) => {
        const [x, y, v, correct, total, wilson] = p.data;
        const name = `<b style="color:${t.fg}">${models[y].name}</b> · ${benches[x].display}`;
        return v == null
          ? `${name}<br/>not attempted`
          : `${name}<br/>${v}% (${correct}/${total}) ±${wilson}pp`;
      },
    },
    xAxis: {
      type: "category",
      data: benches.map((b) => b.display),
      position: "bottom",
      axisLabel: { ...axisLabel(t), fontSize: 10, interval: 0, rotate: 32 },
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
      splitArea: { show: false },
    },
    yAxis: {
      type: "category",
      data: models.map((m) => m.name),
      axisLabel: { ...axisLabel(t), color: t.fg },
      axisLine: { show: false },
      axisTick: { show: false },
      splitArea: { show: false },
    },
    visualMap: {
      min: 0,
      max: 100,
      orient: "horizontal",
      left: "center",
      bottom: 56,
      itemWidth: 10,
      itemHeight: 90,
      text: ["100", "0"],
      textStyle: { color: t.muted, fontFamily: MONO, fontSize: 10 },
      inRange: { color: ["#f87171", "#f0b429", "#34d399"] },
    },
    series: [
      {
        type: "heatmap",
        data,
        itemStyle: { borderColor: t.base, borderWidth: 3, borderRadius: 4 },
        label: {
          show: models.length <= 8,
          formatter: (p: any) => {
            const v = p.data[2];
            if (v == null) return "{dash|–}";
            return v <= 50 ? `{lo|${v}}` : `${v}`;
          },
          color: "#101418",
          fontWeight: 600,
          fontSize: 10,
          fontFamily: MONO,
          rich: {
            lo: { color: "#f2f4f8", fontWeight: 600, fontSize: 10, fontFamily: MONO },
            dash: { color: t.muted, fontWeight: 400, fontSize: 10, fontFamily: MONO },
          },
        },
        emphasis: { itemStyle: { shadowBlur: 12, shadowColor: "rgba(0,0,0,0.4)" } },
      },
    ],
  };
}

function tokensOption(t: Theme): Opt {
  const models = summary.models.filter((m) => m.tokens.total > 0);
  const seg = (key: "input" | "thinking" | "output", name: string, color: string, last = false) => ({
    name,
    type: "bar" as const,
    stack: "tokens",
    barMaxWidth: 30,
    itemStyle: { color, borderRadius: last ? [3, 3, 0, 0] : 0 },
    data: models.map((m) => m.tokens[key]),
  });
  return {
    grid: { left: 52, right: 12, top: 16, bottom: 64 },
    tooltip: {
      ...tooltip(t),
      trigger: "axis",
      axisPointer: { type: "shadow", shadowStyle: { color: t.grid } },
      valueFormatter: (v: any) => kFmt(Number(v)),
    },
    legend: { ...legend(t), data: ["Input", "Thinking", "Output"] },
    xAxis: {
      type: "category",
      data: models.map((m) => m.name),
      axisLabel: { ...axisLabel(t), interval: 0, rotate: models.length > 3 ? 14 : 0 },
      axisLine: { lineStyle: { color: t.axis } },
      axisTick: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: { ...axisLabel(t), formatter: (v: number) => kFmt(v) },
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { show: false },
    },
    series: [
      seg("input", "Input", "#60a5fa"),
      seg("thinking", "Thinking", "#a78bfa"),
      seg("output", "Output", "#2dd4bf", true),
    ],
  };
}

function thinkingOption(t: Theme): Opt {
  const pts = scored;
  return {
    grid: { left: 48, right: 120, top: 20, bottom: 56 },
    tooltip: {
      ...tooltip(t),
      trigger: "item",
      formatter: (p: any) => {
        const [think, score, name] = p.data;
        return `<b style="color:${t.fg}">${name}</b><br/>score ${score.toFixed(1)} · ${kFmt(think)} thinking tok`;
      },
    },
    xAxis: {
      type: "value",
      name: "thinking tokens per run",
      nameLocation: "middle",
      nameGap: 34,
      nameTextStyle: { color: t.muted, fontFamily: MONO, fontSize: 11 },
      axisLabel: { ...axisLabel(t), formatter: (v: number) => kFmt(v) },
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { lineStyle: { color: t.axis } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 100,
      axisLabel: axisLabel(t),
      splitLine: { lineStyle: { color: t.grid } },
      axisLine: { show: false },
    },
    series: [
      {
        type: "scatter",
        symbolSize: 13,
        data: pts.map((m) => ({
          value: [m.tokens.thinking, (m.overall ?? 0) * 100, m.name],
          itemStyle: { color: modelColor(summary.models.indexOf(m)), opacity: 0.9 },
          label: {
            show: true,
            position: "right",
            formatter: () => m.name,
            color: t.muted,
            fontSize: 10,
            fontFamily: MONO,
          },
        })),
        emphasis: { scale: 1.3 },
      },
    ],
  };
}

const builders: Record<string, (t: Theme) => Opt> = {
  cost: costOption,
  radar: radarOption,
  catbars: catbarsOption,
  heatmap: heatmapOption,
  tokens: tokensOption,
  thinking: thinkingOption,
};

const instances = new Map<HTMLElement, echarts.ECharts>();
const built = new Set<HTMLElement>();
let current: Theme = theme(true);

function build(el: HTMLElement) {
  const builder = builders[el.dataset.chart ?? ""];
  if (!builder) return;
  let chart = instances.get(el);
  if (!chart) {
    chart = echarts.init(el);
    instances.set(el, chart);
  }
  chart.setOption(builder(current), true);
  built.add(el);
}

export function initCharts() {
  current = theme(document.documentElement.classList.contains("dark"));
  const els = Array.from(document.querySelectorAll<HTMLElement>("[data-chart]"));

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          build(e.target as HTMLElement);
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: "140px" },
  );
  els.forEach((el) => io.observe(el));

  const ro = new ResizeObserver((entries) => {
    for (const e of entries) instances.get(e.target as HTMLElement)?.resize();
  });
  els.forEach((el) => ro.observe(el));

  window.addEventListener("lb-theme", (ev) => {
    current = theme((ev as CustomEvent<boolean>).detail);
    built.forEach((el) => build(el));
  });
}

export { MODEL_COLORS };
