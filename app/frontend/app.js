/* ==========================================================================
   app.js — interface controller.

   No framework, no build step. Talks to the FastAPI backend when one is
   running and falls back to FI.demo (demo-data.js) when it is not, so the
   page is always presentable.
   ========================================================================== */
(function () {
  "use strict";

  /* When opened straight off disk there is no server to be same-origin with,
     so aim at the default uvicorn address instead. */
  var API_BASE = location.protocol === "file:" || location.port === "5500"
    ? "http://127.0.0.1:8000"
    : "";

  var ATTRS = [
    { key: "item_type", label: "Item type", task: "T1" },
    { key: "season",    label: "Season",    task: "T2" },
    { key: "gender",    label: "Gender",    task: "T3" },
    { key: "usage",     label: "Occasion",  task: "T3" }
  ];

  var state = {
    mode: "idle",      // idle | live | demo
    file: null,        // File or null
    key: null,         // stable identifier used to seed the demo
    objectUrl: null,
    busy: false,
    k: 12,
    prediction: null,
    results: null
  };
  var testSampleIds = null;

  /* ------------------------------------------------------------ helpers */

  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function pct(x, dp) {
    return (x * 100).toFixed(dp === undefined ? 1 : dp) + "%";
  }

  function normalizePrediction(value) {
    if (Array.isArray(value)) return value;
    if (!value) return [];
    var ranked = value.top3 && value.top3.length ? value.top3 : [value];
    return ranked.map(function (r) {
      return { label: r.label, p: r.confidence };
    });
  }

  function normalizeResult(value) {
    return {
      id: value.id,
      similarity: value.similarity,
      article_type: value.articleType || value.article_type,
      base_colour: value.baseColour || value.base_colour,
      image_url: value.image_url
    };
  }

  /* Confidence and similarity live on different scales, so they get
     different cut-points rather than one shared ramp. */
  function tier(v, good, warn) {
    return v >= good ? "good" : v >= warn ? "warn" : "low";
  }
  var confTier = function (v) { return tier(v, 0.85, 0.60); };
  var simTier  = function (v) { return tier(v, 0.88, 0.75); };

  function icon(name, cls) {
    return '<svg class="ic ' + (cls || "") + '" aria-hidden="true"><use href="#ic-' + name + '"/></svg>';
  }

  /* A flat colour swatch stands in for a thumbnail the server cannot serve.
     Tinting it by the item's base colour keeps the tile informative. */
  function swatch(colour) {
    var hex = FI.colourHex[colour] || "#CFCAC1";
    /* Flip the glyph to dark on pale swatches, or White and Cream lose it. */
    var r = parseInt(hex.slice(1, 3), 16),
        g = parseInt(hex.slice(3, 5), 16),
        b = parseInt(hex.slice(5, 7), 16);
    var lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    var ink = lum > 0.65 ? "rgba(0,0,0,.38)" : "rgba(255,255,255,.8)";
    return '<span class="tile-swatch" style="background:' + hex + ';color:' + ink + '">' +
           icon("photo") + "</span>";
  }

  function withTimeout(promise, ms) {
    return new Promise(function (resolve, reject) {
      var t = setTimeout(function () { reject(new Error("timeout")); }, ms);
      promise.then(
        function (v) { clearTimeout(t); resolve(v); },
        function (e) { clearTimeout(t); reject(e); }
      );
    });
  }

  /* --------------------------------------------------------------- mode */

  function setMode(mode, reason) {
    state.mode = mode;
    var pill = $("mode-pill");
    var label = $("mode-label");

    pill.className = "pill pill-" + (mode === "live" ? "live" : mode === "demo" ? "demo" : "idle");
    label.textContent = mode === "live" ? "Live model" : mode === "demo" ? "Demo data" : "Connecting…";
    pill.title = mode === "live"
      ? "Connected to the model server at " + (API_BASE || location.origin)
      : mode === "demo"
        ? "No model server reachable — figures on this page are synthetic"
        : "Looking for a model server…";

    if (mode === "demo") {
      showBanner(
        "No model server is reachable, so this page is showing <strong>synthetic figures</strong> " +
        "for layout purposes. Start the backend and reload to run the real models. " +
        (reason ? "<span class=\"src-note\">(" + esc(reason) + ")</span>" : "")
      );
    } else {
      hideBanner();
    }
  }

  function showBanner(html) {
    $("banner-text").innerHTML = html;
    $("banner").hidden = false;
  }
  function hideBanner() { $("banner").hidden = true; }

  /* The server publishes a model-card row only for tasks whose checkpoint is
     actually loaded. Merge those over the static copy by id, so a live task
     can never advertise demo-data.js numbers while the others keep theirs. */
  function mergeMetrics(live) {
    if (!live || !live.tasks) return;
    if (live.source) FI.metrics.source = live.source;
    live.tasks.forEach(function (row) {
      for (var i = 0; i < FI.metrics.tasks.length; i++) {
        if (FI.metrics.tasks[i].id === row.id) { FI.metrics.tasks[i] = row; return; }
      }
      FI.metrics.tasks.push(row);
    });
  }

  /* A reachable server with a dead model is the dangerous case: the page stays
     in "live" mode and would quietly show nothing for that attribute. Say so. */
  function reportDeadModels(info) {
    var models = (info && info.models) || {};
    var dead = Object.keys(models).filter(function (k) { return models[k].loaded === false; });
    if (!dead.length) { hideBanner(); return; }
    var detail = dead.map(function (k) {
      var why = String(models[k].error || "not loaded");
      if (why.length > 140) why = why.slice(0, 140) + "…";
      return "<b>" + esc(k) + "</b> (" + esc(why) + ")";
    }).join(", ");
    showBanner("Backend is up but these models failed to load: " + detail +
               ". Their predictions are unavailable.");
  }

  function detectBackend() {
    return withTimeout(fetch(API_BASE + "/api/health", { method: "GET" }), 1800)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (info) {
        setMode("live");
        /* Let the server be the authority on labels and published metrics. */
        if (info && info.meta) Object.assign(FI.meta, info.meta);
        if (info && info.metrics) mergeMetrics(info.metrics);
        reportDeadModels(info);
        renderModelCard();
      })
      .catch(function (err) {
        setMode("demo", err && err.message === "timeout" ? "no response within 1.8s" : "connection refused");
      });
  }

  /* ------------------------------------------------------------ rendering */

  function renderPredictions() {
    var host = $("pred-grid");
    var lat = $("latency");

    if (state.busy) {
      host.setAttribute("aria-busy", "true");
      lat.textContent = "";
      host.innerHTML = ATTRS.map(function (a) {
        return '<div class="attr">' +
          '<div class="attr-label">' + esc(a.label) + '<span class="attr-task">' + a.task + "</span></div>" +
          '<span class="skel skel-name"></span>' +
          '<div class="bar"><span style="width:0"></span></div>' +
          "</div>";
      }).join("");
      return;
    }

    host.setAttribute("aria-busy", "false");

    if (!state.prediction) {
      lat.textContent = "";
      host.innerHTML = ATTRS.map(function (a) {
        return '<div class="attr attr-empty">' +
          '<div class="attr-label">' + esc(a.label) + '<span class="attr-task">' + a.task + "</span></div>" +
          '<div class="attr-value"><span class="attr-name">&mdash;</span></div>' +
          '<div class="bar"><span style="width:0"></span></div>' +
          "</div>";
      }).join("");
      return;
    }

    var p = state.prediction;
    lat.textContent = p.latency_ms == null ? "inference complete" : "inference " + p.latency_ms + " ms";

    host.innerHTML = ATTRS.map(function (a) {
      var ranked = (p.predictions[a.key] || []).slice();
      if (!ranked.length) return "";
      var top = ranked[0];
      var t = confTier(top.p);
      var alts = ranked.slice(1, 4);

      var altHtml = alts.length
        ? '<details class="alts"><summary>' + icon("chevron") + "alternatives</summary>" +
            '<div class="alt-list">' +
              alts.map(function (r) {
                return '<div class="alt-row"><b>' + esc(r.label) + "</b><span>" + pct(r.p) + "</span></div>";
              }).join("") +
            "</div></details>"
        : "";

      var fam = (a.key === "item_type" && p.item_family) ? p.item_family : null;
      var famHtml = fam
        ? '<div class="attr-family">category <b>' + esc(fam.label) + "</b> " +
            '<span class="chip chip-' + confTier(fam.confidence) + '">' + pct(fam.confidence) + "</span></div>"
        : "";

      return '<div class="attr">' +
        '<div class="attr-label">' + esc(a.label) + '<span class="attr-task">' + a.task + "</span></div>" +
        '<div class="attr-value">' +
          '<span class="attr-name" title="' + esc(top.label) + '">' + esc(top.label) + "</span>" +
          '<span class="chip chip-' + t + '">' + pct(top.p) + "</span>" +
        "</div>" +
        famHtml +
        '<div class="bar bar-' + t + '" role="progressbar" aria-label="' + esc(a.label) + ' confidence"' +
          ' aria-valuenow="' + Math.round(top.p * 100) + '" aria-valuemin="0" aria-valuemax="100">' +
          '<span style="width:' + Math.max(top.p * 100, 2).toFixed(1) + '%"></span>' +
        "</div>" +
        altHtml +
        "</div>";
    }).join("");
  }

  function renderResults() {
    var host = $("results");

    if (state.busy) {
      host.innerHTML = '<div class="result-grid">' +
        Array.from({ length: state.k }, function () {
          return '<div class="tile">' +
                 '<span class="skel skel-thumb"></span>' +
                 '<span class="skel skel-line" style="width:60%"></span>' +
                 '<span class="skel skel-line" style="width:44%"></span>' +
                 "</div>";
        }).join("") + "</div>";
      return;
    }

    if (!state.results || !state.results.results.length) {
      host.innerHTML = '<div class="empty">' + icon("search", "ic-lg") +
        "<p>Upload an item to retrieve its nearest neighbours from the catalogue.</p></div>";
      return;
    }

    var live = state.mode === "live";
    host.innerHTML = '<div class="result-grid">' +
      state.results.results.map(function (r, i) {
        var t = simTier(r.similarity);
        var thumb = live
          ? '<img src="' + API_BASE + (r.image_url || "") + '" alt="" loading="lazy"' +
            ' data-colour="' + esc(r.base_colour || "") + '">'
          : swatch(r.base_colour);

        return '<figure class="tile">' +
          '<div class="tile-thumb">' + thumb + "</div>" +
          '<figcaption>' +
            '<div class="tile-rank">#' + (i + 1) + " · id " + esc(r.id) + "</div>" +
            '<div class="tile-name" title="' + esc(r.article_type || "") + '">' + esc(r.article_type || "—") + "</div>" +
            '<div class="tile-sub">' + esc(r.base_colour || "") + "</div>" +
          "</figcaption>" +
          '<span class="chip chip-' + t + '">' + pct(r.similarity) + "</span>" +
          "</figure>";
      }).join("") + "</div>";

    /* Any catalogue image the server cannot produce degrades to a colour
       swatch rather than a broken-image glyph. */
    host.querySelectorAll(".tile-thumb img").forEach(function (img) {
      img.addEventListener("error", function () {
        img.parentNode.innerHTML = swatch(img.dataset.colour);
      });
    });
  }

  function renderModelCard() {
    var m = FI.metrics;
    $("model-card-body").innerHTML = m.tasks.map(function (t) {
      return '<div class="mc-row">' +
        '<div class="mc-task">' +
          '<span class="mc-task-id">' + esc(t.id) + "</span>" +
          '<span class="mc-task-name">' + t.name + "</span>" +
        "</div>" +
        '<div class="mc-head">' +
          '<span class="mc-headline">' + t.headline.toFixed(1) + "%</span>" +
          '<span class="mc-headline-label">' + esc(t.headlineLabel) + "</span>" +
        "</div>" +
        '<div>' +
          '<div class="mc-detail">' + t.detail + "</div>" +
          '<span class="mc-flag mc-flag-' + t.flag + '">' + esc(t.flagText) + "</span>" +
          '<div class="mc-detail" style="margin-top:5px;color:var(--text-muted)">' + esc(t.note) + "</div>" +
        "</div>" +
        "</div>";
    }).join("");
    $("mc-source").innerHTML = "Figures read from <code>" + esc(m.source) + "</code>. " +
      "All are held-out test results, not training scores.";
  }

  function renderAll() {
    renderPredictions();
    renderResults();
  }

  /* -------------------------------------------------------------- actions */

  function setImage(file, key, previewSrc) {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;

    state.file = file || null;
    state.key = key;

    var src = previewSrc;
    if (!src && file) {
      state.objectUrl = URL.createObjectURL(file);
      src = state.objectUrl;
    }

    var img = $("preview-img");
    img.onload = function () {
      $("preview-dims").textContent = img.naturalWidth + " × " + img.naturalHeight;
    };
    img.src = src;
    $("preview-name").textContent = file ? file.name : key;
    $("preview-dims").textContent = "";
    $("preview").hidden = false;
    $("dropzone").hidden = true;

    run();
  }

  function clearImage() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    state.file = null;
    state.key = null;
    state.prediction = null;
    state.results = null;
    $("preview").hidden = true;
    $("dropzone").hidden = false;
    hideBanner();
    if (state.mode === "demo") setMode("demo");
    renderAll();
  }

  function run() {
    if (!state.key) return;
    state.busy = true;
    renderAll();

    var work;
    if (state.mode === "live" && state.file) {
      var fd = new FormData();
      fd.append("image", state.file);
      work = withTimeout(
        fetch(API_BASE + "/api/analyze?k=" + encodeURIComponent(state.k) + "&search_mode=nobg", { method: "POST", body: fd }).then(function (r) {
          if (!r.ok) throw new Error("server returned HTTP " + r.status);
          return r.json();
        }),
        20000
      ).then(function (data) {
        var predictions = data.predictions || {};
        return {
          prediction: {
            latency_ms: data.latency_ms,
            /* Task 1 also returns the subCategory it rolls up to. Worth showing:
               on a shifted photograph the family is right 66.95% of the time
               against articleType's 54.86%, because most errors are within-family
               (Casual Shoes for Sports Shoes) and the roll-up absorbs them. */
            item_family: (predictions.articleType || {}).family || null,
            predictions: {
              item_type: normalizePrediction(predictions.articleType),
              season: normalizePrediction(predictions.season),
              gender: normalizePrediction(predictions.gender),
              usage: normalizePrediction(predictions.usage)
            }
          },
          results: {
            results: ((data.visual_search && data.visual_search.similar_items) || [])
              .map(normalizeResult)
          }
        };
      });
    } else {
      /* Offline path — brief delay so the loading state is actually visible. */
      work = new Promise(function (resolve) {
        setTimeout(function () {
          resolve({
            prediction: FI.demo.predict(state.key),
            results: FI.demo.search(state.key, state.k)
          });
        }, 260);
      });
    }

    work.then(function (out) {
      state.prediction = out.prediction;
      state.results = out.results;
    }).catch(function (err) {
      setMode("demo", err && err.message ? err.message : "request failed");
      state.prediction = FI.demo.predict(state.key);
      state.results = FI.demo.search(state.key, state.k);
    }).then(function () {
      state.busy = false;
      renderAll();
    });
  }

  function acceptFile(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) {
      showBanner("That file is not an image. Upload a JPG, PNG or WEBP.");
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      showBanner("That image is larger than 12&nbsp;MB. Try a smaller file.");
      return;
    }
    hideBanner();
    setImage(file, file.name + ":" + file.size);
  }

  function useSample() {
    var ids = testSampleIds || FI.samples.map(function (sample) { return sample.id; });
    var id = ids[Math.floor(Math.random() * ids.length)];
    var s = { id: id, label: "Sample · " + id };
    var url = API_BASE + "/api/catalogue/" + s.id + "/image";

    var samplesRequest = testSampleIds
      ? Promise.resolve({ ids: testSampleIds })
      : fetch(API_BASE + "/api/test-samples").then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        });

    withTimeout(samplesRequest.then(function (data) {
      if (data.ids && data.ids.length) testSampleIds = data.ids;
      var sampleId = testSampleIds[Math.floor(Math.random() * testSampleIds.length)];
      s = { id: sampleId, label: "Sample · " + sampleId };
      url = API_BASE + "/api/catalogue/" + s.id + "/image";
      return fetch(url);
    }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.blob();
    }), 3000)
      .then(function (blob) {
        setImage(new File([blob], s.id + ".jpg", { type: blob.type || "image/jpeg" }), "sample:" + s.id);
      })
      .catch(function () {
        /* No server to fetch the real thumbnail from — use a placeholder so
           the rest of the interface still demonstrates properly. */
        var ph = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(
          '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="160">' +
          '<rect width="120" height="160" fill="#EFEBE4"/>' +
          '<path d="M75 32l30 10v25h-15v45H30V67H15V42l30-10a15 15 0 0 0 30 0z" ' +
          'fill="none" stroke="#B8B2A8" stroke-width="4" stroke-linejoin="round"/></svg>'
        );
        setImage(null, "sample:" + s.id, ph);
        $("preview-name").textContent = s.label + " (placeholder)";
      });
  }

  /* --------------------------------------------------------------- wiring */

  function init() {
    var dz = $("dropzone");
    var input = $("file-input");

    dz.addEventListener("click", function () { input.click(); });
    dz.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });

    input.addEventListener("change", function () {
      acceptFile(input.files && input.files[0]);
      input.value = "";
    });

    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("is-over"); });
    });
    dz.addEventListener("drop", function (e) {
      acceptFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
    });

    /* Keep the browser from navigating away when a drop misses the target. */
    window.addEventListener("dragover", function (e) { e.preventDefault(); });
    window.addEventListener("drop", function (e) { e.preventDefault(); });

    window.addEventListener("paste", function (e) {
      var items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      for (var i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image/") === 0) {
          acceptFile(items[i].getAsFile());
          break;
        }
      }
    });

    $("btn-clear").addEventListener("click", clearImage);
    $("btn-rerun").addEventListener("click", run);
    $("btn-sample").addEventListener("click", useSample);

    document.querySelectorAll(".seg-btn").forEach(function (b) {
      b.addEventListener("click", function () {
        document.querySelectorAll(".seg-btn").forEach(function (o) { o.classList.remove("is-on"); });
        b.classList.add("is-on");
        state.k = Number(b.dataset.k);
        if (state.key) run(); else renderResults();
      });
    });

    renderModelCard();
    renderAll();
    detectBackend();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
