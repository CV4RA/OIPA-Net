"use strict";

const chart = document.querySelector("#chart");
const legend = document.querySelector("#legend");
const filters = Object.fromEntries(
  ["dataset", "protocol", "direction", "metric"].map(id => [id, document.querySelector("#" + id)])
);
const variants = [
  { source: "Zero-shot", label: "Zero-shot", key: "--zero" },
  { source: "Residual heads + MS", label: "Residual heads + MS", key: "--heads" },
  { source: "Q/V LoRA + heads + MS", label: "Q/V LoRA + residual heads + MS", key: "--lora" }
];
const backbones = ["CLIP", "BLIP", "SigLIP 2", "PE-Core"];

function make(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function makeBar(value, variant, backbone, metric, maximum) {
  const row = make("div", "bar-row");
  row.setAttribute("role", "img");
  row.setAttribute("aria-label", backbone + " " + variant.label + ", Recall@" + metric + ": " + value.toFixed(2) + "%");
  const track = make("div", "bar-track");
  const bar = make("div", "bar");
  bar.style.setProperty("--value", (100 * value / maximum) + "%");
  bar.style.setProperty("--key", "var(" + variant.key + ")");
  track.append(bar);
  row.append(track, make("span", "bar-value", value.toFixed(2)));
  return row;
}

function render(rows) {
  const { dataset, protocol, direction, metric } = filters;
  const geographic = protocol.options[1];
  const p1 = dataset.value === "SF-XL-derived" ? "P1-region" : "P1";
  geographic.textContent = p1;
  geographic.value = p1;
  const selected = rows.filter(row => row[0] === dataset.value && row[1] === protocol.value && row[4] === direction.value);
  const col = { "1": 5, "5": 6, "10": 7 }[metric.value];
  const maximum = Math.ceil(Math.max(...rows.map(row => Number(row[col]))) / 10) * 10;
  const groups = backbones.map(backbone => {
    const group = make("div", "plot-group");
    const bars = make("div", "bars");
    for (const variant of variants) {
      const row = selected.find(item => item[2] === backbone && item[3] === variant.source);
      if (row) bars.append(makeBar(Number(row[col]), variant, backbone, metric.value, maximum));
    }
    group.append(make("span", "backbone", backbone), bars);
    return group;
  });
  const axis = make("div", "plot-axis");
  axis.append(make("span", "", "0%"), make("span", "", (maximum / 2) + "%"), make("span", "", maximum + "%"));
  chart.replaceChildren(...groups, axis);
}

async function load() {
  const response = await fetch("matched_full_gallery_recall.csv");
  if (!response.ok) throw new Error("Archive unavailable");
  const rows = (await response.text()).trim().split(/\r?\n/).slice(1).map(line => line.split(","));
  if (rows.some(row => row.length !== 8)) throw new Error("Unexpected CSV schema");
  for (const variant of variants) {
    const item = make("span", "", variant.label);
    item.style.setProperty("--key", "var(" + variant.key + ")");
    legend.append(item);
  }
  for (const input of Object.values(filters)) input.addEventListener("change", () => render(rows));
  render(rows);
}

load().catch(() => {
  chart.textContent = "The result table could not be loaded. Use the CSV download to view the measurements.";
});
