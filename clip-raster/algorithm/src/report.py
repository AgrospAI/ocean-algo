import base64
import io
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.image import imsave
from numpy.ma import MaskedArray
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import box, mapping

from .geometry import Parcel
from .indices import INDEXES
from .raster import Grid, project
from .scenes import Scene

WEB_MERCATOR = "EPSG:3857"
HISTOGRAM_BINS = 20
SIGPAC_CODE = ("provincia", "municipio", "agregado", "zona", "poligono", "parcela", "recinto")


def statistics(values: MaskedArray, parcel_pixels: int) -> dict[str, float | int | None]:
    data = values.compressed()
    if data.size == 0:
        return {"validPixels": 0, "validShare": 0.0}
    p10, median, p90 = np.percentile(data, [10, 50, 90])
    return {
        "mean": round(float(data.mean()), 4),
        "median": round(float(median), 4),
        "p10": round(float(p10), 4),
        "p90": round(float(p90), 4),
        "std": round(float(data.std()), 4),
        "min": round(float(data.min()), 4),
        "max": round(float(data.max()), 4),
        "validPixels": int(data.size),
        "validShare": round(data.size / parcel_pixels, 4) if parcel_pixels else 0.0,
    }


def histogram(values: MaskedArray, vmin: float, vmax: float) -> list[int]:
    counts, _ = np.histogram(values.compressed(), bins=HISTOGRAM_BINS, range=(vmin, vmax))
    return counts.tolist()


def sigpac_code(properties: dict[str, Any]) -> str | None:
    if not all(key in properties for key in SIGPAC_CODE):
        return None
    return ":".join(str(properties[key]) for key in SIGPAC_CODE)


def summary(
    parcel: Parcel,
    grid: Grid,
    scene: Scene,
    indices: dict[str, MaskedArray],
    lookback_days: int,
    max_cloud_cover: float,
) -> dict[str, Any]:
    """Machine-readable result: small enough for an LLM to read through the MCP."""
    props = scene.item["properties"]
    parcel_pixels = int(grid.inside.sum())
    area_m2 = project(parcel.geometry, "EPSG:4326", grid.crs).area
    centroid = parcel.geometry.centroid
    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "parcel": {
            "sigpac": [sigpac_code(p) for p in parcel.properties if sigpac_code(p)],
            "attributes": parcel.properties,
            "areaHa": round(area_m2 / 10_000, 4),
            "centroid": [round(centroid.x, 6), round(centroid.y, 6)],
            "pixels": parcel_pixels,
        },
        "scene": {
            "id": scene.id,
            "datetime": scene.datetime,
            "platform": props.get("platform"),
            "tile": props.get("grid:code"),
            "tileCloudCover": props.get("eo:cloud_cover"),
            "parcelCloudShare": round(scene.cloud_fraction, 4),
            "parcelCoverage": round(scene.coverage, 4),
            "meetsCloudThreshold": scene.meets_threshold,
            "source": "Earth Search (Element84), collection sentinel-2-c1-l2a",
        },
        "parameters": {"lookbackDays": lookback_days, "maxCloudCover": max_cloud_cover},
        "indices": {
            index.name: {"formula": index.formula, **statistics(indices[index.name], parcel_pixels)}
            for index in INDEXES
        },
    }


def to_web_mercator(array: np.ndarray, grid: Grid, resampling: Resampling) -> tuple:
    """Reproject a (bands, rows, cols) array so Leaflet can overlay it without distortion."""
    bands = array.shape[0]
    left, bottom, right, top = array_bounds(grid.height, grid.width, grid.transform)
    dst_transform, width, height = calculate_default_transform(
        grid.crs, WEB_MERCATOR, grid.width, grid.height, left, bottom, right, top
    )
    out = np.full((bands, height, width), np.nan, dtype="float32")
    reproject(
        array.astype("float32"),
        out,
        src_transform=grid.transform,
        src_crs=grid.crs,
        src_nodata=np.nan,
        dst_transform=dst_transform,
        dst_crs=WEB_MERCATOR,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return out, array_bounds(height, width, dst_transform)


def latlon_bounds(bounds: tuple[float, float, float, float]) -> list[list[float]]:
    minx, miny, maxx, maxy = project(box(*bounds), WEB_MERCATOR, "EPSG:4326").bounds
    return [[miny, minx], [maxy, maxx]]


def png_data_uri(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    imsave(buffer, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def index_overlay(values: MaskedArray, grid: Grid, colormap: str, vmin: float, vmax: float):
    filled = np.ma.filled(values.astype("float32"), np.nan)[np.newaxis]
    projected, bounds = to_web_mercator(filled, grid, Resampling.nearest)
    data = projected[0]
    rgba = colormaps[colormap](Normalize(vmin, vmax, clip=True)(np.nan_to_num(data)))
    rgba[..., 3] = np.where(np.isnan(data), 0, 1)
    return png_data_uri(rgba), latlon_bounds(bounds)


def true_color_overlay(rgb: np.ndarray, grid: Grid):
    projected, bounds = to_web_mercator(rgb, grid, Resampling.nearest)
    alpha = ~np.isnan(projected).any(axis=0) & (np.nan_to_num(projected).sum(axis=0) > 0)
    rgba = np.dstack([np.nan_to_num(band) / 255 for band in projected] + [alpha.astype(float)])
    return png_data_uri(np.clip(rgba, 0, 1)), latlon_bounds(bounds)


def gradient(colormap: str, steps: int = 11) -> list[str]:
    cmap = colormaps[colormap]
    return [
        "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in cmap(i / (steps - 1))[:3]))
        for i in range(steps)
    ]


def render_html(
    result: dict[str, Any],
    parcel: Parcel,
    grid: Grid,
    indices: dict[str, MaskedArray],
    true_color: np.ndarray | None,
) -> str:
    layers = []
    if true_color is not None:
        image, bounds = true_color_overlay(true_color, grid)
        layers.append({"id": "rgb", "label": "Color real", "image": image, "bounds": bounds})
    for index in INDEXES:
        image, bounds = index_overlay(
            indices[index.name], grid, index.colormap, index.vmin, index.vmax
        )
        layers.append(
            {
                "id": index.name,
                "label": index.label,
                "image": image,
                "bounds": bounds,
                "vmin": index.vmin,
                "vmax": index.vmax,
                "gradient": gradient(index.colormap),
                "formula": index.formula,
                "meaning": index.meaning,
                "histogram": histogram(indices[index.name], index.vmin, index.vmax),
            }
        )

    data = {
        "summary": result,
        "layers": layers,
        "parcel": mapping(parcel.geometry),
    }
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    codes = result["parcel"]["sigpac"]
    title = f"Recinto SIGPAC {', '.join(codes)}" if codes else "Parcela"
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


TEMPLATE = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Índices Sentinel-2 · __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root{--bg:#f6f7f5;--card:#fff;--ink:#1f2a1f;--muted:#5f6b5f;--line:#dde3dc;--accent:#2e7d32;--warn:#8a5a00;--warnbg:#fff4d6}
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
  header{padding:18px 24px;background:var(--card);border-bottom:1px solid var(--line)}
  header h1{margin:0 0 4px;font-size:20px}
  header .sub{color:var(--muted);font-size:14px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
  .chip{background:var(--bg);border:1px solid var(--line);border-radius:999px;padding:4px 10px;font-size:13px}
  .chip b{font-weight:600}
  .warn{margin:12px 24px 0;padding:10px 14px;border-radius:8px;background:var(--warnbg);color:var(--warn);font-size:14px}
  main{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:16px;padding:16px 24px 24px;align-items:start}
  @media (max-width:960px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
  #mapcard{padding:0;overflow:hidden;display:flex;flex-direction:column;position:sticky;top:16px}
  @media (max-width:960px){#mapcard{position:static}}
  .tabs{display:flex;flex-wrap:wrap;gap:6px;padding:10px 12px;border-bottom:1px solid var(--line)}
  .tabs button{border:1px solid var(--line);background:var(--bg);border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
  .tabs button.on{background:var(--accent);border-color:var(--accent);color:#fff}
  .tabs label{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted)}
  #map{height:540px;width:100%}
  .leaflet-image-layer{image-rendering:pixelated}
  .legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);font-size:12px;min-width:180px}
  .legend .bar{height:10px;border-radius:3px;margin:4px 0}
  .legend .ends{display:flex;justify-content:space-between;color:var(--muted)}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
  .idx h3{margin:0;font-size:16px;display:flex;justify-content:space-between;align-items:baseline}
  .idx .mean{font-size:26px;font-weight:700}
  .idx .formula{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
  .idx p{font-size:13px;color:var(--muted);margin:8px 0 0}
  .idx svg{width:100%;height:46px;margin-top:8px}
  table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
  th,td{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
  th{color:var(--muted);font-weight:500}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  h2{font-size:15px;margin:0 0 8px}
  section+section{margin-top:16px}
  footer{padding:0 24px 24px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>Índices de vegetación Sentinel-2 · __TITLE__</h1>
  <div class="sub" id="sub"></div>
  <div class="chips" id="chips"></div>
</header>
<div id="warnings"></div>
<main>
  <div class="card" id="mapcard">
    <div class="tabs" id="tabs">
      <label>Opacidad <input type="range" id="opacity" min="0" max="100" value="85"></label>
    </div>
    <div id="map"></div>
  </div>
  <div>
    <section><div class="grid" id="cards"></div></section>
    <section class="card"><h2>Atributos SIGPAC</h2><div id="attrs"></div></section>
  </div>
</main>
<footer id="footer"></footer>
<script>
const DATA = __DATA__;
const S = DATA.summary;
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const pct = v => (v === null || v === undefined) ? "—" : `${(v * 100).toFixed(0)} %`;
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

const date = new Date(S.scene.datetime);
document.getElementById("sub").textContent =
  `Escena del ${date.toLocaleDateString("es-ES", {day: "numeric", month: "long", year: "numeric"})} · ` +
  `${S.parcel.areaHa.toFixed(2)} ha · ${S.parcel.pixels} píxeles de 10 m`;
const chips = [
  ["Satélite", (S.scene.platform || "").toUpperCase()],
  ["Tile", S.scene.tile],
  ["Nubes en la parcela", pct(S.scene.parcelCloudShare)],
  ["Nubes en el tile", S.scene.tileCloudCover == null ? "—" : `${S.scene.tileCloudCover.toFixed(0)} %`],
  ["Búsqueda", `últimos ${S.parameters.lookbackDays} días, ≤ ${S.parameters.maxCloudCover} % nubes`],
];
document.getElementById("chips").innerHTML =
  chips.map(([k, v]) => `<span class="chip">${esc(k)}: <b>${esc(v ?? "—")}</b></span>`).join("");

const warnings = [];
if (!S.scene.meetsCloudThreshold)
  warnings.push(`Ninguna escena de los últimos ${S.parameters.lookbackDays} días tiene ≤ ${S.parameters.maxCloudCover} % de nubes sobre la parcela; se muestra la más despejada (${pct(S.scene.parcelCloudShare)} de píxeles descartados).`);
if (S.scene.parcelCoverage < 0.999)
  warnings.push(`La escena solo cubre el ${pct(S.scene.parcelCoverage)} de la parcela.`);
document.getElementById("warnings").innerHTML = warnings.map(w => `<div class="warn">${esc(w)}</div>`).join("");

const map = L.map("map", {zoomSnap: 0.25});
const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {maxZoom: 20, maxNativeZoom: 19, attribution: "© OpenStreetMap"});
const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {maxZoom: 20, maxNativeZoom: 19, attribution: "Esri World Imagery"}).addTo(map);
L.control.layers({"Ortofoto": sat, "OpenStreetMap": osm}, null, {position: "topleft"}).addTo(map);
const outline = L.geoJSON(DATA.parcel, {style: {color: "#ffffff", weight: 2, fill: false, dashArray: "4 3"}});
map.fitBounds(outline.getBounds(), {padding: [30, 30]});

const legend = L.control({position: "bottomright"});
legend.onAdd = () => L.DomUtil.create("div", "legend");
legend.addTo(map);

const overlays = {};
let active = null;
const opacity = document.getElementById("opacity");
for (const layer of DATA.layers) {
  overlays[layer.id] = L.imageOverlay(layer.image, layer.bounds, {opacity: opacity.value / 100});
  const button = document.createElement("button");
  button.textContent = layer.label;
  button.dataset.id = layer.id;
  button.onclick = () => show(layer.id);
  document.getElementById("tabs").insertBefore(button, opacity.parentElement);
}
function show(id) {
  if (active) map.removeLayer(overlays[active]);
  active = id;
  overlays[id].addTo(map);
  outline.remove(); outline.addTo(map);
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.id === id));
  const layer = DATA.layers.find(l => l.id === id);
  const box = document.querySelector(".legend");
  if (layer.gradient) {
    const st = S.indices[id];
    box.innerHTML = `<b>${esc(layer.label)}</b> · media ${fmt(st.mean)}` +
      `<div class="bar" style="background:linear-gradient(to right,${layer.gradient.join(",")})"></div>` +
      `<div class="ends"><span>${layer.vmin}</span><span>${(layer.vmin + layer.vmax) / 2}</span><span>${layer.vmax}</span></div>`;
    box.style.display = "";
  } else {
    box.style.display = "none";
  }
}
opacity.oninput = () => Object.values(overlays).forEach(o => o.setOpacity(opacity.value / 100));
show(DATA.layers.find(l => l.id === "ndvi") ? "ndvi" : DATA.layers[0].id);

function histogramSvg(layer) {
  const counts = layer.histogram, max = Math.max(1, ...counts), w = 100 / counts.length;
  return `<svg viewBox="0 0 100 40" preserveAspectRatio="none">` + counts.map((c, i) => {
    const color = layer.gradient[Math.round(i / (counts.length - 1) * (layer.gradient.length - 1))];
    const h = c / max * 38;
    return `<rect x="${i * w + 0.3}" y="${40 - h}" width="${w - 0.6}" height="${h}" fill="${color}"/>`;
  }).join("") + `</svg>`;
}
document.getElementById("cards").innerHTML = DATA.layers.filter(l => l.gradient).map(layer => {
  const st = S.indices[layer.id];
  const rows = [["Mediana", st.median], ["P10 – P90", null, `${fmt(st.p10)} – ${fmt(st.p90)}`],
                ["Desv. típica", st.std], ["Mín – máx", null, `${fmt(st.min)} – ${fmt(st.max)}`],
                ["Píxeles válidos", null, `${st.validPixels} (${pct(st.validShare)})`]];
  return `<div class="card idx" data-id="${layer.id}">
    <h3><span>${esc(layer.label)}</span><span class="mean">${fmt(st.mean, 2)}</span></h3>
    <div class="formula">${esc(layer.formula)}</div>
    ${histogramSvg(layer)}
    <table>${rows.map(([k, v, t]) => `<tr><th>${k}</th><td class="num">${t ?? fmt(v)}</td></tr>`).join("")}</table>
    <p>${esc(layer.meaning)}</p></div>`;
}).join("");
document.querySelectorAll(".idx").forEach(card => { card.style.cursor = "pointer"; card.onclick = () => show(card.dataset.id); });

const attrs = S.parcel.attributes;
if (attrs.length) {
  const keys = [...new Set(attrs.flatMap(Object.keys))];
  document.getElementById("attrs").innerHTML = `<table>` +
    keys.map(k => `<tr><th>${esc(k)}</th>${attrs.map(a => `<td>${esc(a[k] ?? "—")}</td>`).join("")}</tr>`).join("") +
    `</table>`;
} else {
  document.getElementById("attrs").textContent = "El dataset no incluye atributos.";
}
document.getElementById("footer").innerHTML =
  `Contiene datos modificados de Copernicus Sentinel ${date.getUTCFullYear()} (${esc(S.scene.id)}), ` +
  `obtenidos de ${esc(S.scene.source)}. Píxeles con nubes, sombras, nieve o sin datos excluidos según la capa SCL. ` +
  `Generado por AgrospAI Clip Raster el ${new Date(S.generatedAt).toLocaleString("es-ES")}.`;
</script>
</body>
</html>
"""
