# Frontend

The demonstration interface for the Fashion Intelligence models: upload a garment
photo, read the four predicted attributes with their confidences, and browse the
nearest catalogue items from the retrieval index.

## Why plain HTML instead of React

- The repository is otherwise entirely Python. `requirements.txt` declares no JS
  toolchain and there is no `package.json`, so a React/Vite build would make the
  submission depend on Node as well as Python just to open one page.
- The interface is a single screen with four pieces of state (file, predictions,
  results, error). React's benefits — routing, component reuse, shared state —
  do not apply; its costs (build step, `node_modules`, CORS between `:5173` and
  `:8000`) do.
- FastAPI can serve this folder as static files on its own origin, which removes
  CORS configuration entirely.

Nothing here is compiled, minified, or fetched from a CDN. Every icon is an inline
SVG symbol, so the page also works with no network at all.

## Files

| File | Role |
| --- | --- |
| `index.html` | Page structure and the SVG icon sprite |
| `styles.css` | All styling; light and dark palettes as CSS custom properties |
| `app.js` | Controller — upload handling, API calls, rendering |
| `demo-data.js` | Label vocabularies, published metrics, offline stand-in |

## Running it

**With a backend** (the intended path). `app/backend/main.py` already mounts this
folder as static files on the API's own origin, so there is nothing to wire up:

```python
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")  # mount last
```

Run from the PROJECT ROOT, not from `app/frontend`:

```
python -m uvicorn app.backend.main:app --reload
```

Then open <http://127.0.0.1:8000/>. The status pill reads *Live model* once
`/api/health` answers.

**Without a backend.** Double-click `index.html`, or serve the folder with
`python -m http.server 5500` from `app/frontend`. The page probes `/api/health`,
finds nothing, switches the status pill to *Demo data*, and shows a banner saying
the figures are synthetic. Useful for working on layout; not evidence of anything.

When opened over `file://` — or from port 5500 — the page targets
`http://127.0.0.1:8000` for the API, so a separately running backend is still
picked up. That path needs CORS, which the backend enables.

**What to upload.** Images from `A2_FashionDataset/FashionDataset/train/images_train`
are the distribution the models were trained on — white-background product
cutouts — and item type will typically come back above 90%. The unlabelled
`test/images_test` folder is mostly on-model lifestyle photography and sits a long
way off that distribution (`unlabelled_distribution_shift_tvd: 44.6` in
`artifacts/task1/task1_summary.json`), so low, split confidences there are the
honest output, not a serving fault.

## API contract

Four endpoints are used by the page. All JSON is `application/json`. The backend
also exposes `POST /api/task{1,2,3}/predict` and `POST /api/task4/search` for
scoring one model at a time; the page does not call them, `/api/analyze` covers
the whole pipeline in one request.

### `GET /api/health`

Probed once on load to decide live vs. demo mode. `meta` and `metrics` are
optional; when present they override the values baked into `demo-data.js`, which
lets the server stay the single source of truth for labels and metrics. The page
reads them straight from the loaded checkpoints, so the model card on screen can
never advertise a stale run.

`models` is what drives the "backend is up but a model failed to load" banner.

```jsonc
{
  "status": "ok",
  "meta": { "itemType": ["Accessory Gift Set", "Backpacks", "..."] },
  "metrics": { "source": "artifacts/…", "tasks": [ /* see demo-data.js */ ] },
  "models": {
    "task1": { "loaded": true, "classes": 92, "device": "cpu", "error": null },
    "task2": { "loaded": true, "classes": 4, "device": "cpu", "error": null },
    "task3": { "loaded": true, "classes": { "gender": 5, "usage": 4 }, "device": "cpu", "error": null },
    "task4": { "loaded": true, "catalogue_size": 32837, "embedding_dim": 128,
               "method": "Improved+TTA+bgaug", "device": "cpu", "error": null }
  }
}
```

### `POST /api/analyze`

`multipart/form-data` with `image` (the file). Two query parameters: `k`, how many
neighbours to return (default 10, clamped to 1–20), and `search_mode`, one of
`letterbox` / `crop` / `nobg` (default `nobg`). One call runs all four classifiers
and the retrieval index, so the page makes a single request per upload.

Every prediction is a `{ label, confidence, top3 }` object, **already sorted
descending** — `top3[0]` is the answer and the rest are the *alternatives*.
`confidence` is a probability in `[0, 1]`, displayed as a percentage. `app.js`
flattens these to its own `{ label, p }` shape in `normalizePrediction`, so the
camelCase keys below are what the wire actually carries.

```jsonc
{
  "filename": "52003.jpg",
  "latency_ms": 118.4,
  "predictions": {
    "articleType": {
      "label": "Tshirts",
      "confidence": 0.2353,
      "top3": [
        { "label": "Tshirts", "confidence": 0.2353 },
        { "label": "Jeans", "confidence": 0.2075 },
        { "label": "Sports Shoes", "confidence": 0.0974 }
      ]
    },
    "season": { "label": "Fall", "confidence": 0.5668, "top3": [ /* … */ ] },
    "gender": { "label": "Men",  "confidence": 0.9992, "top3": [ /* … */ ] },
    "usage":  { "label": "Casual", "confidence": 1.0,  "top3": [ /* … */ ] }
  },
  "visual_search": {
    "method": "Improved+TTA+bgaug",
    "mode": "nobg",
    "k": 3,
    "similar_items": [
      {
        "rank": 1,
        "id": 15007,
        "articleType": "Jackets",
        "subCategory": "Topwear",
        "masterCategory": "Apparel",
        "baseColour": "Black",
        "gender": "Men",
        "usage": "Sports",
        "productDisplayName": "ADIDAS Originals Men Solid Black Jackets",
        "similarity": 0.6811,
        "image_url": "/api/catalogue/15007/image"
      }
    ]
  },
  "backend_stage": "all_tasks_connected"
}
```

`similarity` is cosine similarity in `[0, 1]`, displayed as a percentage.
`latency_ms` covers the four models plus retrieval; when it is absent the page
shows "inference complete" instead of a figure.

### `GET /api/catalogue/{item_id}/image`

Returns the catalogue JPEG for a product id, used for result thumbnails and the
*Use a sample item* button — it is the path each result's `image_url` points at.
If this 404s, tiles fall back to a swatch tinted by the item's `baseColour`, so it
is optional, but the page looks considerably better with it.

### `GET /api/test-samples`

```jsonc
{ "ids": [52003, 52004, "…"] }
```

Ids the backend can serve a real thumbnail for, used to populate the sample
buttons. The page falls back to the static list in `demo-data.js` if this fails.

## Design notes

- **Confidence is never hidden.** Every prediction carries a percentage chip and a
  bar, coloured green above 85%, amber above 60%, clay below. On catalogue-style
  product shots item type usually reads green; on lifestyle photography it drops
  to amber or clay, which is the honest result for that input.
- **Alternatives are one click away** rather than buried. Season is right 67.5% of
  the time and confuses adjacent seasons constantly, and item type degrades sharply
  off-distribution — in both cases the runner-up list is where the model's actual
  belief lives.
- **The model card sits on the page**, not in a footnote. A reading of "Fall, 57%"
  means something different once you know that model is right 67.5% of the time
  against a 49.6% majority-class baseline, and the page says so. Its numbers come
  from `/api/health` when a backend is up, so they track the deployed checkpoints.
- **Demo mode announces itself** in the status pill and a banner. Synthetic
  numbers are never presented as if a model produced them.
- Dark mode follows `prefers-color-scheme`; both palettes are token swaps, so
  layout cannot diverge between them.
