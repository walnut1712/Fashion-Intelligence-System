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

**With a backend** (the intended path). Serve this folder from FastAPI so the page
and the API share an origin, then open <http://127.0.0.1:8000/>:

```python
# app/backend/main.py
from pathlib import Path
from fastapi.staticfiles import StaticFiles

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")  # mount last
```

```
python -m uvicorn app.backend.main:app --reload
```

**Without a backend.** Double-click `index.html`, or serve the folder with
`python -m http.server 5500` from `app/frontend`. The page probes `/api/health`,
finds nothing, switches the status pill to *Demo data*, and shows a banner saying
the figures are synthetic. Useful for working on layout; not evidence of anything.

When opened over `file://` the page targets `http://127.0.0.1:8000` for the API,
so a separately running backend is still picked up — that path needs CORS enabled.

## API contract

Three endpoints. All JSON is `application/json`.

### `GET /api/health`

Probed once on load to decide live vs. demo mode. Both extra keys are optional;
when present they override the values baked into `demo-data.js`, which lets the
server stay the single source of truth for labels and metrics.

```jsonc
{
  "status": "ok",
  "meta": {
    "itemType": ["Backpacks", "Belts", "..."],
    "season":   ["Fall", "Spring", "Summer", "Winter"],
    "gender":   ["Boys", "Girls", "Men", "Unisex", "Women"],
    "usage":    ["Casual", "Ethnic", "Formal", "Sports"]
  },
  "metrics": { "source": "artifacts/…", "tasks": [ /* see demo-data.js */ ] }
}
```

### `POST /api/analyse`

`multipart/form-data` with `image` (the file) and `k` (how many neighbours to
return). One call runs all four classifiers and the retrieval index, so the page
makes a single request per upload.

```jsonc
{
  "latency_ms": 42,
  "predictions": {
    "item_type": [{ "label": "Jackets", "p": 0.31 }, { "label": "Sweaters", "p": 0.18 }],
    "season":    [{ "label": "Winter",  "p": 0.72 }],
    "gender":    [{ "label": "Men",     "p": 0.93 }],
    "usage":     [{ "label": "Casual",  "p": 0.88 }]
  },
  "results": [
    { "id": 19283, "similarity": 0.947, "article_type": "Jackets", "base_colour": "Navy Blue" }
  ]
}
```

Each `predictions` array must be **sorted descending by `p`**. The page shows
element 0 as the answer and elements 1–3 under *alternatives*. Returning at least
the top 3 matters for item type, where top-1 accuracy is 20.8% but top-3 is 37.2%
— the runner-ups carry most of the usable signal.

`similarity` is cosine similarity in `[0, 1]`, displayed as a percentage.

### `GET /api/image/{id}`

Returns the catalogue JPEG for a product id, used for result thumbnails and the
*Use a sample item* button. If this 404s, tiles fall back to a swatch tinted by
the item's `base_colour` — so it is optional, but the page looks considerably
better with it.

## Design notes

- **Confidence is never hidden.** Every prediction carries a percentage chip and a
  bar, coloured green above 85%, amber above 60%, clay below. Item type will
  usually render amber or clay, which is the honest result for that model.
- **Alternatives are one click away** rather than buried, because top-1 is weak
  for item type and season, and the runner-up list is where the model's actual
  belief lives.
- **The model card sits on the page**, not in a footnote. A reading of "Jackets,
  31%" means something different once you know that model is right 20.8% of the
  time against an 18.6% baseline, and the page says so.
- **Demo mode announces itself** in the status pill and a banner. Synthetic
  numbers are never presented as if a model produced them.
- Dark mode follows `prefers-color-scheme`; both palettes are token swaps, so
  layout cannot diverge between them.
