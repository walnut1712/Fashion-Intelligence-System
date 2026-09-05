#!/usr/bin/env python
"""Produce a labelling sheet for the real user photographs.

Why this exists
---------------
Every Task 4 number is measured on something other than the thing the system is
for. ``P@10 80.2`` is catalogue photographs retrieving catalogue photographs;
``P@10 60.6`` is catalogue items composited onto procedural backgrounds. The 31
photographs in ``A2_FashionDataset/input_images`` are the only real uploads in
the project, and they carry no labels at all - so the domain the encoder is
being improved for has never been scored, only described.

Labelling 31 images by hand is a twenty-minute job, and it is the difference
between "similarity drops from 0.837 to 0.664 on real photographs" and "P@10 on
real photographs is X". This writes two things:

    A2_FashionDataset/input_images_labels.csv   the template, one row per photo
    outputs/input_images_labelling.html         a self-contained sheet showing
                                                each photo with type-ahead inputs
                                                and a button that emits the CSV

The sheet is the working copy; the CSV is the deliverable. Nothing here scores
anything - ``src/evaluation/metrics.py`` does that once the labels exist.

Usage
-----
    python scripts/build_label_sheet.py
    python scripts/build_label_sheet.py --force   # overwrite an existing CSV
"""

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = PROJECT_ROOT / "A2_FashionDataset" / "input_images"
GALLERY = PROJECT_ROOT / "artifacts" / "task4" / "gallery_metadata.csv"
LABELS_CSV = PROJECT_ROOT / "A2_FashionDataset" / "input_images_labels.csv"
SHEET_HTML = PROJECT_ROOT / "outputs" / "input_images_labelling.html"

COLUMNS = ["file", "articleType", "baseColour", "n_garments", "notes"]
THUMB_MAX = (320, 320)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=UPLOAD_DIR)
    parser.add_argument("--force", action="store_true",
                        help="overwrite input_images_labels.csv if it already "
                             "has labels in it")
    return parser.parse_args()


def thumbnail(path):
    """A PNG data URI, or None when the format has no decoder installed.

    The uploads are deliberately a mess of formats - .webp, .avif, .png, .jpg -
    because that is what a real upload is. An .avif needs a Pillow plugin that
    may not be present, and a missing thumbnail should cost that one row, not
    the sheet.
    """
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail(THUMB_MAX, Image.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
    except Exception as error:                       # noqa: BLE001 - reported, not raised
        print("  no thumbnail for {}: {}".format(path.name, error))
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_sheet(rows, types, colours):
    cards = []
    for index, (name, uri) in enumerate(rows):
        picture = ('<img src="{}" alt="">'.format(uri) if uri
                   else '<div class="missing">no preview<br>open the file directly</div>')
        cards.append("""
    <figure class="card">
      <div class="thumb">{picture}</div>
      <figcaption>
        <div class="name" title="{name}">{name}</div>
        <label>Type <input list="types" data-field="articleType" data-row="{index}" autocomplete="off"></label>
        <label>Colour <input list="colours" data-field="baseColour" data-row="{index}" autocomplete="off"></label>
        <label>Garments <input type="number" min="0" step="1" value="1" data-field="n_garments" data-row="{index}"></label>
        <label>Notes <input data-field="notes" data-row="{index}" autocomplete="off"></label>
      </figcaption>
    </figure>""".format(picture=picture, name=html.escape(name), index=index))

    return """<!doctype html>
<meta charset="utf-8">
<title>Label the real uploads</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; margin: 0; padding: 24px; }}
  h1 {{ font-size: 19px; margin: 0 0 4px; }}
  p.lede {{ margin: 0 0 20px; max-width: 78ch; opacity: .8; }}
  .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }}
  .card {{ margin: 0; border: 1px solid #8884; border-radius: 10px; overflow: hidden; }}
  .thumb {{ aspect-ratio: 3/4; display: grid; place-items: center; background: #8881; }}
  .thumb img {{ max-width: 100%; max-height: 100%; }}
  .missing {{ font-size: 12px; text-align: center; opacity: .7; padding: 12px; }}
  figcaption {{ padding: 10px 12px 12px; display: grid; gap: 6px; }}
  .name {{ font-size: 11px; opacity: .65; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  label {{ display: grid; grid-template-columns: 74px 1fr; align-items: center; gap: 6px; font-size: 12px; }}
  input {{ font: inherit; font-size: 12px; padding: 4px 6px; border: 1px solid #8886; border-radius: 6px; background: transparent; color: inherit; min-width: 0; }}
  .bar {{ position: sticky; bottom: 0; margin-top: 22px; padding: 14px 0; background: Canvas; border-top: 1px solid #8884; }}
  button {{ font: inherit; padding: 8px 14px; border-radius: 8px; border: 1px solid #8886; background: transparent; color: inherit; cursor: pointer; }}
  textarea {{ width: 100%; height: 190px; margin-top: 10px; font-family: ui-monospace, monospace; font-size: 12px; }}
</style>

<h1>Label the real uploads</h1>
<p class="lede">
  These {count} photographs are the only real-domain data in the project, and nothing
  measures retrieval on them because they carry no ground truth. Type is the one that
  matters - it is what P@10 scores. Colour is the control attribute. Set
  <b>Garments</b> to the number of separate items visible, and 0 if the photo is not
  clothing at all, so full outfits and non-clothing can be scored separately rather
  than counted as failures. Then press the button and save the CSV over
  <code>A2_FashionDataset/input_images_labels.csv</code>.
</p>

<datalist id="types">{types}</datalist>
<datalist id="colours">{colours}</datalist>

<div class="grid">{cards}
</div>

<div class="bar">
  <button id="emit">Build the CSV</button>
  <textarea id="out" readonly placeholder="The CSV appears here."></textarea>
</div>

<script>
  const FILES = {files};
  document.getElementById("emit").addEventListener("click", function () {{
    const rows = FILES.map(function (name, i) {{
      const get = function (field) {{
        const el = document.querySelector('[data-row="' + i + '"][data-field="' + field + '"]');
        return el ? el.value.trim() : "";
      }};
      return [name, get("articleType"), get("baseColour"), get("n_garments"), get("notes")];
    }});
    const quote = function (v) {{
      return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
    }};
    const blank = rows.filter(function (r) {{ return !r[1]; }}).length;
    document.getElementById("out").value =
      (blank ? "# " + blank + " row(s) still have no type\\n" : "") +
      {header} + "\\n" + rows.map(function (r) {{ return r.map(quote).join(","); }}).join("\\n");
  }});
</script>
""".format(
        count=len(rows),
        cards="".join(cards),
        files=json.dumps([name for name, _ in rows]),
        header=json.dumps(",".join(COLUMNS)),
        types="".join('<option value="{}">'.format(html.escape(t)) for t in types),
        colours="".join('<option value="{}">'.format(html.escape(c)) for c in colours),
    )


def main():
    args = parse_args()
    if not args.images.is_dir():
        sys.exit("Upload directory not found: {}".format(args.images))

    files = sorted(p for p in args.images.iterdir() if p.is_file())
    if not files:
        sys.exit("No images in {}".format(args.images))
    print("Uploads: {}".format(len(files)))

    if not GALLERY.exists():
        sys.exit("Cannot read the label vocabulary without {}".format(GALLERY))
    gallery = pd.read_csv(GALLERY)
    types = sorted(gallery["articleType"].dropna().unique())
    colours = sorted(gallery["baseColour"].dropna().unique())
    print("Vocabulary: {} types, {} colours".format(len(types), len(colours)))

    rows = [(path.name, thumbnail(path)) for path in files]

    if LABELS_CSV.exists() and not args.force:
        existing = pd.read_csv(LABELS_CSV)
        if existing.get("articleType", pd.Series(dtype=object)).notna().any():
            print("{} already has labels - left alone (use --force to replace)".format(
                LABELS_CSV.name))
        else:
            pd.DataFrame({"file": [n for n, _ in rows]}).reindex(
                columns=COLUMNS).to_csv(LABELS_CSV, index=False)
    else:
        pd.DataFrame({"file": [n for n, _ in rows]}).reindex(
            columns=COLUMNS).to_csv(LABELS_CSV, index=False)
        print("Wrote {}".format(LABELS_CSV))

    SHEET_HTML.parent.mkdir(parents=True, exist_ok=True)
    SHEET_HTML.write_text(build_sheet(rows, types, colours), encoding="utf-8")
    print("Wrote {}".format(SHEET_HTML))
    print("\nOpen the sheet, label the photographs, then save the CSV it produces over"
          "\n  {}".format(LABELS_CSV))


if __name__ == "__main__":
    main()
