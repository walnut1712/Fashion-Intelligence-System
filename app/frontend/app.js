/* ==========================================================================
   app.js — interface controller.

   No framework, no build step. Talks to the FastAPI backend when one is
   running and falls back to FI.demo (demo-data.js) when it is not.
   ========================================================================== */

(function () {
  "use strict";

  var API_BASE =
    location.protocol === "file:" || location.port === "5500"
      ? "http://127.0.0.1:8000"
      : "";

  var ATTRS = [
    { key: "item_type", label: "Item type", task: "T1" },
    { key: "season", label: "Season", task: "T2" },
    { key: "gender", label: "Gender", task: "T3" },
    { key: "usage", label: "Occasion", task: "T3" }
  ];

  var state = {
    mode: "idle",
    file: null,
    key: null,
    objectUrl: null,
    busy: false,
    k: 12,
    view: "whole",     // whole | regions
    prediction: null,
    results: null,
    regions: null
  };

  var testSampleIds = null;

  /* ------------------------------------------------------------ helpers */

  var $ = function (id) {
    return document.getElementById(id);
  };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[c];
    });
  }

  function pct(x, dp) {
    return (
      (x * 100).toFixed(
        dp === undefined ? 1 : dp
      ) + "%"
    );
  }

  function probability01(value) {
    var n = Number(value);

    if (!isFinite(n)) {
      return NaN;
    }

    if (n > 1 && n <= 100) {
      n = n / 100;
    }

    return Math.max(
      0,
      Math.min(1, n)
    );
  }

  /*
    Standard model prediction normaliser.

    Accepts:
      [{label, p}, ...]
      {label, confidence}
      {top3: [...]}
  */
  function normalizePrediction(value) {
    if (Array.isArray(value)) {
      return value.map(function (r) {
        return {
          label: r.label,
          p: probability01(
            r.p != null
              ? r.p
              : r.confidence
          )
        };
      });
    }

    if (!value) {
      return [];
    }

    var ranked =
      value.top3 &&
      value.top3.length
        ? value.top3
        : [value];

    return ranked.map(function (r) {
      return {
        label: r.label,
        p: probability01(
          r.p != null
            ? r.p
            : r.confidence
        )
      };
    });
  }

  /*
    TASK 2 PRESENTATION

    Primary:
      Task2A original four-class CNN:
      Spring / Summer / Fall / Winter.

    Auxiliary:
      Task2B SFS recommendation.

    Task2B NEVER replaces the main Task2A output.
    If Task2B has no defensible recommendation,
    the recommendation line is simply omitted.
  */
  function normalizeSeasonPrediction(
    value,
    recommendation
  ) {
    var ranked =
      normalizePrediction(value);

    if (!ranked.length) {
      return [];
    }

    var recommendationLabel = null;

    if (
      recommendation &&
      recommendation.recommendation_available === true &&
      recommendation.display_label
    ) {
      recommendationLabel =
        recommendation.display_label;
    }

    /*
      Put auxiliary information only on
      the top Task2A prediction object.
    */
    ranked[0].is_season_primary = true;

    ranked[0].recommendation_label =
      recommendationLabel;

    ranked[0].recommendation_rule =
      recommendationLabel &&
      recommendation
        ? recommendation.display_rule
        : null;

    ranked[0].recommendation_source =
      recommendationLabel &&
      recommendation
        ? recommendation.recommendation_source
        : null;

    return ranked;
  }

  /*
    Task3B only adds a small fine-grained recommendation.
    It NEVER replaces the primary Task3 Usage prediction.
  */
  function normalizeUsagePrediction(
    value,
    recommendation
  ) {
    var ranked =
      normalizePrediction(value);

    if (!ranked.length) {
      return [];
    }

    var output =
      ranked.map(
        function (row) {
          return {
            label: row.label,
            p: row.p
          };
        }
      );

    if (
      recommendation &&
      recommendation.recommendation_available === true &&
      recommendation.display_label
    ) {
      output[0].recommendation_label =
        recommendation.display_label;
    }

    return output;
  }


  function normalizeResult(value) {
    return {
      id: value.id,

      similarity:
        probability01(
          value.similarity
        ),

      article_type:
        value.articleType ||
        value.article_type,

      base_colour:
        value.baseColour ||
        value.base_colour,

      image_url:
        value.image_url
    };
  }

  /* The service decides whether it trusts a retrieval - top-1 similarity
     against 0.70, neighbour coherence against 0.50, thresholds taken from
     catalogue photographs scoring 0.837/0.833 against real uploads at
     0.664/0.489. That verdict used to be fetched and dropped here, so the one
     thing the system knew about its own weakest case never reached the page. */
  function normalizeDiagnostics(value) {
    if (!value) return null;
    return {
      confident: value.confident !== false,
      top1_similarity: value.top1_similarity,
      coherence: value.coherence,
      ingest_method: value.ingest_method,
      ingest_fell_back: !!value.ingest_fell_back
    };
  }

  function tier(
    v,
    good,
    warn
  ) {
    return v >= good
      ? "good"
      : v >= warn
        ? "warn"
        : "low";
  }

  var confTier = function (v) {
    return tier(
      v,
      0.85,
      0.60
    );
  };

  var simTier = function (v) {
    return tier(
      v,
      0.88,
      0.75
    );
  };

  function icon(
    name,
    cls
  ) {
    return (
      '<svg class="ic ' +
      (cls || "") +
      '" aria-hidden="true">' +
      '<use href="#ic-' +
      name +
      '"/>' +
      "</svg>"
    );
  }

  function swatch(colour) {
    var hex =
      FI.colourHex[colour] ||
      "#CFCAC1";

    var r =
      parseInt(
        hex.slice(1, 3),
        16
      );

    var g =
      parseInt(
        hex.slice(3, 5),
        16
      );

    var b =
      parseInt(
        hex.slice(5, 7),
        16
      );

    var lum =
      (
        0.299 * r +
        0.587 * g +
        0.114 * b
      ) / 255;

    var ink =
      lum > 0.65
        ? "rgba(0,0,0,.38)"
        : "rgba(255,255,255,.8)";

    return (
      '<span class="tile-swatch" ' +
      'style="background:' +
      hex +
      ";color:" +
      ink +
      '">' +
      icon("photo") +
      "</span>"
    );
  }

  function withTimeout(
    promise,
    ms
  ) {
    return new Promise(
      function (
        resolve,
        reject
      ) {
        var t =
          setTimeout(
            function () {
              reject(
                new Error("timeout")
              );
            },
            ms
          );

        promise.then(
          function (v) {
            clearTimeout(t);
            resolve(v);
          },

          function (e) {
            clearTimeout(t);
            reject(e);
          }
        );
      }
    );
  }

  /* --------------------------------------------------------------- mode */

  function setMode(
    mode,
    reason
  ) {
    state.mode = mode;

    var pill =
      $("mode-pill");

    var label =
      $("mode-label");

    pill.className =
      "pill pill-" +
      (
        mode === "live"
          ? "live"
          : mode === "demo"
            ? "demo"
            : "idle"
      );

    label.textContent =
      mode === "live"
        ? "Live model"
        : mode === "demo"
          ? "Demo data"
          : "Connecting…";

    pill.title =
      mode === "live"
        ? (
            "Connected to the model server at " +
            (
              API_BASE ||
              location.origin
            )
          )
        : mode === "demo"
          ? (
              "No model server reachable — " +
              "figures on this page are synthetic"
            )
          : "Looking for a model server…";

    if (mode === "demo") {
      showBanner(
        "No model server is reachable, so this page is showing " +
        "<strong>synthetic figures</strong> for layout purposes. " +
        "Start the backend and reload to run the real models. " +
        (
          reason
            ? (
                '<span class="src-note">(' +
                esc(reason) +
                ")</span>"
              )
            : ""
        )
      );
    } else {
      hideBanner();
    }
  }

  function showBanner(html) {
    $("banner-text").innerHTML =
      html;

    $("banner").hidden =
      false;
  }

  function hideBanner() {
    $("banner").hidden =
      true;
  }

  function mergeMetrics(live) {
    if (
      !live ||
      !live.tasks
    ) {
      return;
    }

    if (live.source) {
      FI.metrics.source =
        live.source;
    }

    live.tasks.forEach(
      function (row) {
        for (
          var i = 0;
          i < FI.metrics.tasks.length;
          i++
        ) {
          if (
            FI.metrics.tasks[i].id ===
            row.id
          ) {
            FI.metrics.tasks[i] =
              row;

            return;
          }
        }

        FI.metrics.tasks.push(
          row
        );
      }
    );
  }

  function reportDeadModels(info) {
    var models =
      (
        info &&
        info.models
      ) || {};

    /*
      Task2B is auxiliary only.
      Its failure must not mark the main app as broken.
    */
    var dead =
      Object.keys(models)
        .filter(function (k) {
          return (
            k !== "task2b" &&
            models[k].loaded === false
          );
        });

    if (!dead.length) {
      hideBanner();
      return;
    }

    var detail =
      dead.map(
        function (k) {
          var why =
            String(
              models[k].error ||
              "not loaded"
            );

          if (
            why.length > 140
          ) {
            why =
              why.slice(
                0,
                140
              ) + "…";
          }

          return (
            "<b>" +
            esc(k) +
            "</b> (" +
            esc(why) +
            ")"
          );
        }
      ).join(", ");

    showBanner(
      "Backend is up but these models failed to load: " +
      detail +
      ". Their predictions are unavailable."
    );
  }

  function detectBackend() {
    return withTimeout(
      fetch(
        API_BASE +
        "/api/health",
        {
          method: "GET"
        }
      ),
      1800
    )
      .then(
        function (r) {
          if (!r.ok) {
            throw new Error(
              "HTTP " +
              r.status
            );
          }

          return r.json();
        }
      )
      .then(
        function (info) {
          setMode("live");

          if (
            info &&
            info.meta
          ) {
            Object.assign(
              FI.meta,
              info.meta
            );
          }

          if (
            info &&
            info.metrics
          ) {
            mergeMetrics(
              info.metrics
            );
          }

          reportDeadModels(
            info
          );

          renderModelCard();
        }
      )
      .catch(
        function (err) {
          setMode(
            "demo",
            err &&
            err.message === "timeout"
              ? "no response within 1.8s"
              : (
                  err &&
                  err.message
                    ? err.message
                    : "connection refused"
                )
          );
        }
      );
  }

  /* ------------------------------------------------------------ rendering */

  function renderPredictions() {
    var host =
      $("pred-grid");

    var lat =
      $("latency");

    if (state.busy) {
      host.setAttribute(
        "aria-busy",
        "true"
      );

      lat.textContent = "";

      host.innerHTML =
        ATTRS.map(
          function (a) {
            return (
              '<div class="attr">' +

                '<div class="attr-label">' +
                  esc(a.label) +
                  '<span class="attr-task">' +
                    a.task +
                  "</span>" +
                "</div>" +

                '<span class="skel skel-name"></span>' +

                '<div class="bar">' +
                  '<span style="width:0"></span>' +
                "</div>" +

              "</div>"
            );
          }
        ).join("");

      return;
    }

    host.setAttribute(
      "aria-busy",
      "false"
    );

    if (!state.prediction) {
      lat.textContent = "";

      host.innerHTML =
        ATTRS.map(
          function (a) {
            return (
              '<div class="attr attr-empty">' +

                '<div class="attr-label">' +
                  esc(a.label) +
                  '<span class="attr-task">' +
                    a.task +
                  "</span>" +
                "</div>" +

                '<div class="attr-value">' +
                  '<span class="attr-name">&mdash;</span>' +
                "</div>" +

                '<div class="bar">' +
                  '<span style="width:0"></span>' +
                "</div>" +

              "</div>"
            );
          }
        ).join("");

      return;
    }

    var p =
      state.prediction;

    lat.textContent =
      p.latency_ms == null
        ? "inference complete"
        : (
            "inference " +
            p.latency_ms +
            " ms"
          );

    host.innerHTML =
      ATTRS.map(
        function (a) {
          var ranked =
            (
              p.predictions[
                a.key
              ] || []
            ).slice();

          if (!ranked.length) {
            return "";
          }

          var top =
            ranked[0];

          /*
            Task3B recommendation:
            only visible for Occasion when backend says
            recommendation_available=true.
          */
          var usageRecommendationHtml =
            (
              a.key === "usage" &&
              top.recommendation_label
            )
              ? (
                  '<div class="attr-family">' +
                    'Also suitable for: <b>' +
                    esc(
                      top.recommendation_label
                    ) +
                    '</b>' +
                  '</div>'
                )
              : "";


          var displayProbability =
            probability01(
              top.p
            );

          var hasDisplayProbability =
            isFinite(
              displayProbability
            );

          var t =
            hasDisplayProbability
              ? confTier(
                  displayProbability
                )
              : "warn";

          /* --------------------------------------------------
             Task2B auxiliary recommendation
             -------------------------------------------------- */

          var recommendationHtml =
            "";

          if (
            a.key === "season" &&
            top.recommendation_label
          ) {
            recommendationHtml =
              (
                '<div class="attr-family">' +
                  "Also suitable for: <b>" +
                    esc(
                      top.recommendation_label
                    ) +
                  "</b>" +
                "</div>"
              );
          }

          /* --------------------------------------------------
             Alternatives
             -------------------------------------------------- */

          var alts =
            ranked.slice(
              1,
              4
            );

          var altHtml =
            alts.length
              ? (
                  '<details class="alts">' +

                    "<summary>" +
                      icon(
                        "chevron"
                      ) +
                      "alternatives" +
                    "</summary>" +

                    '<div class="alt-list">' +

                      alts.map(
                        function (r) {
                          var altP =
                            probability01(
                              r.p
                            );

                          return (
                            '<div class="alt-row">' +

                              "<b>" +
                                esc(
                                  r.label
                                ) +
                              "</b>" +

                              "<span>" +
                                (
                                  isFinite(
                                    altP
                                  )
                                    ? pct(
                                        altP
                                      )
                                    : ""
                                ) +
                              "</span>" +

                            "</div>"
                          );
                        }
                      ).join("") +

                    "</div>" +

                  "</details>"
                )
              : "";

          /* --------------------------------------------------
             Task1 family
             -------------------------------------------------- */

          var fam =
            (
              a.key === "item_type" &&
              p.item_family
            )
              ? p.item_family
              : null;

          var famProbability =
            fam
              ? probability01(
                  fam.confidence
                )
              : NaN;

          var famHtml =
            fam
              ? (
                  '<div class="attr-family">' +

                    "category <b>" +
                      esc(
                        fam.label
                      ) +
                    "</b> " +

                    (
                      isFinite(
                        famProbability
                      )
                        ? (
                            '<span class="chip chip-' +
                              confTier(
                                famProbability
                              ) +
                            '">' +
                              pct(
                                famProbability
                              ) +
                            "</span>"
                          )
                        : ""
                    ) +

                  "</div>"
                )
              : "";

          /* --------------------------------------------------
             Main confidence chip
             -------------------------------------------------- */

          var chipHtml =
            hasDisplayProbability
              ? (
                  '<span class="chip chip-' +
                    t +
                  '">' +
                    pct(
                      displayProbability
                    ) +
                  "</span>"
                )
              : (
                  '<span class="chip chip-warn">—</span>'
                );

          /* --------------------------------------------------
             Main confidence bar
             -------------------------------------------------- */

          var barLabel =
            a.label +
            " confidence";

          var barHtml =
            hasDisplayProbability
              ? (
                  '<div class="bar bar-' +
                    t +
                  '"' +

                  ' role="progressbar"' +

                  ' aria-label="' +
                    esc(
                      barLabel
                    ) +
                  '"' +

                  ' aria-valuenow="' +
                    Math.round(
                      displayProbability *
                      100
                    ) +
                  '"' +

                  ' aria-valuemin="0"' +
                  ' aria-valuemax="100">' +

                    '<span style="width:' +
                      Math.max(
                        displayProbability *
                        100,
                        2
                      ).toFixed(1) +
                    '%"></span>' +

                  "</div>"
                )
              : (
                  '<div class="bar">' +
                    '<span style="width:0"></span>' +
                  "</div>"
                );

          return (
            '<div class="attr">' +

              '<div class="attr-label">' +
                esc(
                  a.label
                ) +
                '<span class="attr-task">' +
                  a.task +
                "</span>" +
              "</div>" +

              '<div class="attr-value">' +

                '<span class="attr-name" title="' +
                  esc(
                    top.label
                  ) +
                '">' +
                  esc(
                    top.label
                  ) +
                "</span>" +

                chipHtml +

              "</div>" +

              famHtml +

              recommendationHtml +

              usageRecommendationHtml +
          barHtml +

              altHtml +

            "</div>"
          );
        }
      ).join("");
  }

  /* ---------------------------------------------------------- search UI */

  /* Shown only when the engine is not confident. A wrong answer presented with
     the same certainty as a right one is the failure the gate exists to
     prevent; the gate is advisory, so every result is still returned and
     ranked and this only says how much to trust the top of the list. */
  function lowConfidenceNotice(d) {
    if (!d || d.confident) return "";

    var reasons = [];
    if (typeof d.top1_similarity === "number" && d.top1_similarity < 0.70) {
      reasons.push("closest match is only " + pct(d.top1_similarity));
    }
    if (typeof d.coherence === "number" && d.coherence < 0.50) {
      reasons.push("the matches disagree with each other");
    }
    if (d.ingest_fell_back) {
      reasons.push("the background could not be separated from the item");
    }

    return '<div class="banner banner-inline">' + icon("alert") +
      "<span><b>Low confidence in these matches.</b> " +
      (reasons.length ? esc(reasons.join("; ")) + ". " : "") +
      "The catalogue is flat-lay product shots on white, so a photograph taken " +
      "in the wild is harder than the published accuracy suggests.</span></div>";
  }

  /* propose_regions names bands by geometry; a reader wants body parts. */
  var REGION_LABELS = {
    whole: "Whole image", upper: "Upper body", lower: "Lower body",
    top3: "Top third", mid3: "Middle third", low3: "Bottom third"
  };

  function regionLabel(name) {
    if (REGION_LABELS[name]) return REGION_LABELS[name];
    var piece = /^part(\d+)$/.exec(name);
    return piece ? "Separate piece " + piece[1] : name;
  }

  function tileHtml(r, i) {
    var similarity = probability01(r.similarity);
    var t = simTier(similarity);
    var live = state.mode === "live";
    var thumb = live
      ? '<img src="' + API_BASE + (r.image_url || "") + '" alt="" loading="lazy"' +
        ' data-colour="' + esc(r.base_colour || "") + '">'
      : swatch(r.base_colour);

    return '<figure class="tile">' +
      '<div class="tile-thumb">' + thumb + "</div>" +
      "<figcaption>" +
        '<div class="tile-rank">#' + (i + 1) + " \u00b7 id " + esc(r.id) + "</div>" +
        '<div class="tile-name" title="' + esc(r.article_type || "") + '">' +
          esc(r.article_type || "\u2014") + "</div>" +
        '<div class="tile-sub">' + esc(r.base_colour || "") + "</div>" +
      "</figcaption>" +
      '<span class="chip chip-' + t + '">' + pct(similarity) + "</span>" +
    "</figure>";
  }

  /* Any catalogue image the server cannot produce degrades to a colour swatch
     rather than a broken-image glyph. */
  function bindThumbFallback(host) {
    host.querySelectorAll(".tile-thumb img").forEach(function (img) {
      img.addEventListener("error", function () {
        img.parentNode.innerHTML = swatch(img.dataset.colour);
      });
    });
  }

  function renderRegions(host) {
    var data = state.regions;
    if (!data || !data.regions || !data.regions.length) {
      host.innerHTML = '<div class="empty">' + icon("search", "ic-lg") +
        "<p>No separate garments were found in this photograph.</p></div>";
      return;
    }

    /* The API sorts accepted regions first. Rejected ones are still shown,
       dimmed: "we looked here and were not convinced" is more useful than
       dropping the region, and it is how the acceptance rule can be judged
       rather than trusted. */
    host.innerHTML = data.regions.map(function (r) {
      var tiles = r.items.map(normalizeResult).slice(0, state.k);
      return '<div class="region-group' + (r.accepted ? "" : " region-skipped") + '">' +
        '<div class="region-head">' +
          '<span class="region-name">' + esc(regionLabel(r.region)) + "</span>" +
          '<span class="region-tag' + (r.accepted ? "" : " region-tag-off") + '">' +
            (r.accepted ? "kept" : "not convincing") + "</span>" +
          '<span class="region-meta">closest ' + pct(r.top_similarity) +
            " \u00b7 agreement " + pct(r.coherence) + "</span>" +
        "</div>" +
        '<div class="result-grid">' + tiles.map(tileHtml).join("") + "</div>" +
      "</div>";
    }).join("") +
    '<p class="card-note">Segmentation: ' + esc(data.segmentation || "none") +
      " \u00b7 " + data.accepted_count + " of " + data.regions.length +
      " regions kept.</p>";
  }

  function renderResults() {
    var host =
      $("results");

    if (state.busy) {
      host.innerHTML =
        (
          '<div class="result-grid">' +

            Array.from(
              {
                length:
                  state.k
              },
              function () {
                return (
                  '<div class="tile">' +
                    '<span class="skel skel-thumb"></span>' +
                    '<span class="skel skel-line" style="width:60%"></span>' +
                    '<span class="skel skel-line" style="width:44%"></span>' +
                  "</div>"
                );
              }
            ).join("") +

          "</div>"
        );

      return;
    }

    if (state.view === "regions") {
      if (state.mode !== "live") {
        host.innerHTML = '<div class="empty">' + icon("alert", "ic-lg") +
          "<p>Per-garment search needs the live model; the offline demo has " +
          "no segmentation.</p></div>";
        return;
      }
      renderRegions(host);
      bindThumbFallback(host);
      return;
    }

    if (
      !state.results ||
      !state.results.results.length
    ) {
      host.innerHTML =
        (
          '<div class="empty">' +

            icon(
              "search",
              "ic-lg"
            ) +

            "<p>" +
              "Upload an item to retrieve its nearest neighbours from the catalogue." +
            "</p>" +

          "</div>"
        );

      return;
    }

    var live =
      state.mode === "live";

    host.innerHTML =
      (
        lowConfidenceNotice(
          state.results.diagnostics
        ) +

        '<div class="result-grid">' +

          state
            .results
            .results
            .map(
              function (
                r,
                i
              ) {
                var similarity =
                  probability01(
                    r.similarity
                  );

                var t =
                  simTier(
                    similarity
                  );

                var thumb =
                  live
                    ? (
                        '<img src="' +
                          API_BASE +
                          (
                            r.image_url ||
                            ""
                          ) +
                        '" alt="" loading="lazy"' +

                        ' data-colour="' +
                          esc(
                            r.base_colour ||
                            ""
                          ) +
                        '">'
                      )
                    : swatch(
                        r.base_colour
                      );

                return (
                  '<figure class="tile">' +

                    '<div class="tile-thumb">' +
                      thumb +
                    "</div>" +

                    "<figcaption>" +

                      '<div class="tile-rank">' +
                        "#" +
                        (i + 1) +
                        " · id " +
                        esc(r.id) +
                      "</div>" +

                      '<div class="tile-name" title="' +
                        esc(
                          r.article_type ||
                          ""
                        ) +
                      '">' +
                        esc(
                          r.article_type ||
                          "—"
                        ) +
                      "</div>" +

                      '<div class="tile-sub">' +
                        esc(
                          r.base_colour ||
                          ""
                        ) +
                      "</div>" +

                    "</figcaption>" +

                    '<span class="chip chip-' +
                      t +
                    '">' +
                      pct(
                        similarity
                      ) +
                    "</span>" +

                  "</figure>"
                );
              }
            )
            .join("") +

        "</div>"
      );

    host
      .querySelectorAll(
        ".tile-thumb img"
      )
      .forEach(
        function (img) {
          img.addEventListener(
            "error",
            function () {
              img.parentNode
                .innerHTML =
                  swatch(
                    img.dataset
                      .colour
                  );
            }
          );
        }
      );
  }

  /* ---------------------------------------------------------- model card */

  function renderModelCard() {
    var m =
      FI.metrics;

    $("model-card-body")
      .innerHTML =
        m.tasks
          .map(
            function (t) {
              return (
                '<div class="mc-row">' +

                  '<div class="mc-task">' +

                    '<span class="mc-task-id">' +
                      esc(t.id) +
                    "</span>" +

                    '<span class="mc-task-name">' +
                      t.name +
                    "</span>" +

                  "</div>" +

                  '<div class="mc-head">' +

                    '<span class="mc-headline">' +
                      t.headline.toFixed(1) +
                    "%</span>" +

                    '<span class="mc-headline-label">' +
                      esc(
                        t.headlineLabel
                      ) +
                    "</span>" +

                  "</div>" +

                  "<div>" +

                    '<div class="mc-detail">' +
                      t.detail +
                    "</div>" +

                    '<span class="mc-flag mc-flag-' +
                      t.flag +
                    '">' +
                      esc(
                        t.flagText
                      ) +
                    "</span>" +

                    '<div class="mc-detail" ' +
                      'style="margin-top:5px;color:var(--text-muted)">' +
                      /* Same treatment as t.detail above. Both are authored by
                         us - demo-data.js, or a service's model_card() - and
                         both use <b> for the figure that matters. Escaping only
                         this one printed the tags literally: "P@10 falls to
                         <b>60.6</b>". */
                      t.note +
                    "</div>" +

                  "</div>" +

                "</div>"
              );
            }
          )
          .join("");

    $("mc-source")
      .innerHTML =
        (
          "Figures read from <code>" +
          esc(
            m.source
          ) +
          "</code>. " +
          "All are held-out test results, not training scores."
        );
  }

  function renderAll() {
    renderPredictions();
    renderResults();
  }

  /* -------------------------------------------------------------- actions */

  function setImage(
    file,
    key,
    previewSrc
  ) {
    if (
      state.objectUrl
    ) {
      URL.revokeObjectURL(
        state.objectUrl
      );
    }

    state.objectUrl =
      null;

    state.file =
      file || null;

    state.key =
      key;

    var src =
      previewSrc;

    if (
      !src &&
      file
    ) {
      state.objectUrl =
        URL.createObjectURL(
          file
        );

      src =
        state.objectUrl;
    }

    var img =
      $("preview-img");

    img.onload =
      function () {
        $("preview-dims")
          .textContent =
            (
              img.naturalWidth +
              " × " +
              img.naturalHeight
            );
      };

    img.src =
      src;

    $("preview-name")
      .textContent =
        file
          ? file.name
          : key;

    $("preview-dims")
      .textContent = "";

    $("preview")
      .hidden = false;

    $("dropzone")
      .hidden = true;

    run();
  }

  function clearImage() {
    if (
      state.objectUrl
    ) {
      URL.revokeObjectURL(
        state.objectUrl
      );
    }

    state.objectUrl =
      null;

    state.file =
      null;

    state.key =
      null;

    state.prediction =
      null;

    state.results =
      null;

    $("preview")
      .hidden = true;

    $("dropzone")
      .hidden = false;

    hideBanner();

    if (
      state.mode === "demo"
    ) {
      setMode(
        "demo"
      );
    }

    renderAll();
  }

  function run() {
    if (!state.key) {
      return;
    }

    state.busy =
      true;

    renderAll();

    var work;

    if (
      state.view === "regions" &&
      state.mode === "live" &&
      state.file
    ) {
      var rfd = new FormData();
      rfd.append("image", state.file);

      work = withTimeout(
        fetch(
          API_BASE + "/api/task4/regions?k=" +
            encodeURIComponent(state.k) + "&mode=nobg",
          { method: "POST", body: rfd }
        ).then(function (r) {
          if (!r.ok) {
            throw new Error("server returned HTTP " + r.status);
          }
          return r.json();
        }),
        30000
      ).then(function (data) {
        return {
          prediction: state.prediction,
          results: state.results,
          regions: data
        };
      });

    } else if (
      state.mode === "live" &&
      state.file
    ) {
      var fd =
        new FormData();

      fd.append(
        "image",
        state.file
      );

      work =
        withTimeout(
          fetch(
            API_BASE +
              "/api/analyze?k=" +
              encodeURIComponent(
                state.k
              ) +
              "&search_mode=nobg",
            {
              method:
                "POST",

              body:
                fd
            }
          )
            .then(
              function (r) {
                if (!r.ok) {
                  throw new Error(
                    "server returned HTTP " +
                    r.status
                  );
                }

                return r.json();
              }
            ),
          20000
        )
          .then(
            function (data) {
              var predictions =
                data.predictions ||
                {};

              return {
                prediction: {
                  latency_ms:
                    data.latency_ms,

                  item_family:
                    (
                      predictions
                        .articleType ||
                      {}
                    ).family ||
                    null,

                  predictions: {
                    item_type:
                      normalizePrediction(
                        predictions
                          .articleType
                      ),

                    /*
                      Task2A is primary.
                      Task2B only provides the small
                      auxiliary recommendation line.
                    */
                    season:
                      normalizeSeasonPrediction(
                        predictions
                          .season,
                        predictions
                          .season_recommendation
                      ),

                    gender:
                      normalizePrediction(
                        predictions
                          .gender
                      ),

                    usage:
                      normalizeUsagePrediction(
                        predictions
                          .usage,
                        predictions
                          .usage_recommendation
                      )
                  }
                },

                results: {
                  results:
                    (
                      (
                        data
                          .visual_search &&
                        data
                          .visual_search
                          .similar_items
                      ) ||
                      []
                    ).map(
                      normalizeResult
                    ),

                  diagnostics:
                    normalizeDiagnostics(
                      data
                        .visual_search &&
                      data
                        .visual_search
                        .diagnostics
                    )
                }
              };
            }
          );

    } else {
      work =
        new Promise(
          function (
            resolve
          ) {
            setTimeout(
              function () {
                resolve({
                  prediction:
                    FI.demo.predict(
                      state.key
                    ),

                  results:
                    FI.demo.search(
                      state.key,
                      state.k
                    )
                });
              },
              260
            );
          }
        );
    }

    work
      .then(
        function (out) {
          state.prediction =
            out.prediction;

          state.results =
            out.results;

          state.regions =
            out.regions || null;
        }
      )
      .catch(
        function (err) {
          setMode(
            "demo",
            err &&
            err.message
              ? err.message
              : "request failed"
          );

          state.prediction =
            FI.demo.predict(
              state.key
            );

          state.results =
            FI.demo.search(
              state.key,
              state.k
            );
        }
      )
      .then(
        function () {
          state.busy =
            false;

          renderAll();
        }
      );
  }

  function acceptFile(file) {
    if (!file) {
      return;
    }

    if (
      !/^image\//.test(
        file.type
      )
    ) {
      showBanner(
        "That file is not an image. Upload a JPG, PNG or WEBP."
      );

      return;
    }

    if (
      file.size >
      12 *
      1024 *
      1024
    ) {
      showBanner(
        "That image is larger than 12&nbsp;MB. Try a smaller file."
      );

      return;
    }

    hideBanner();

    setImage(
      file,
      file.name +
        ":" +
        file.size
    );
  }

  function useSample() {
    var ids =
      testSampleIds ||
      FI.samples.map(
        function (sample) {
          return sample.id;
        }
      );

    var id =
      ids[
        Math.floor(
          Math.random() *
          ids.length
        )
      ];

    var s = {
      id: id,

      label:
        "Sample · " +
        id
    };

    var url =
      API_BASE +
      "/api/catalogue/" +
      s.id +
      "/image";

    var samplesRequest =
      testSampleIds
        ? Promise.resolve({
            ids:
              testSampleIds
          })
        : fetch(
            API_BASE +
            "/api/test-samples"
          )
            .then(
              function (r) {
                if (!r.ok) {
                  throw new Error(
                    "HTTP " +
                    r.status
                  );
                }

                return r.json();
              }
            );

    withTimeout(
      samplesRequest
        .then(
          function (data) {
            if (
              data.ids &&
              data.ids.length
            ) {
              testSampleIds =
                data.ids;
            }

            var sampleId =
              testSampleIds[
                Math.floor(
                  Math.random() *
                  testSampleIds.length
                )
              ];

            s = {
              id:
                sampleId,

              label:
                "Sample · " +
                sampleId
            };

            url =
              API_BASE +
              "/api/catalogue/" +
              s.id +
              "/image";

            return fetch(
              url
            );
          }
        )
        .then(
          function (r) {
            if (!r.ok) {
              throw new Error(
                "HTTP " +
                r.status
              );
            }

            return r.blob();
          }
        ),
      3000
    )
      .then(
        function (blob) {
          setImage(
            new File(
              [blob],
              s.id +
                ".jpg",
              {
                type:
                  blob.type ||
                  "image/jpeg"
              }
            ),
            "sample:" +
              s.id
          );
        }
      )
      .catch(
        function () {
          var ph =
            (
              "data:image/svg+xml;charset=utf-8," +
              encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="160">' +
                '<rect width="120" height="160" fill="#EFEBE4"/>' +
                '<path d="M75 32l30 10v25h-15v45H30V67H15V42l30-10a15 15 0 0 0 30 0z" ' +
                'fill="none" stroke="#B8B2A8" stroke-width="4" stroke-linejoin="round"/>' +
                "</svg>"
              )
            );

          setImage(
            null,
            "sample:" +
              s.id,
            ph
          );

          $("preview-name")
            .textContent =
              (
                s.label +
                " (placeholder)"
              );
        }
      );
  }

  /* --------------------------------------------------------------- wiring */

  function init() {
    var dz =
      $("dropzone");

    var input =
      $("file-input");

    dz.addEventListener(
      "click",
      function () {
        input.click();
      }
    );

    dz.addEventListener(
      "keydown",
      function (e) {
        if (
          e.key === "Enter" ||
          e.key === " "
        ) {
          e.preventDefault();

          input.click();
        }
      }
    );

    input.addEventListener(
      "change",
      function () {
        acceptFile(
          input.files &&
          input.files[0]
        );

        input.value =
          "";
      }
    );

    [
      "dragenter",
      "dragover"
    ].forEach(
      function (ev) {
        dz.addEventListener(
          ev,
          function (e) {
            e.preventDefault();

            dz.classList.add(
              "is-over"
            );
          }
        );
      }
    );

    [
      "dragleave",
      "drop"
    ].forEach(
      function (ev) {
        dz.addEventListener(
          ev,
          function (e) {
            e.preventDefault();

            dz.classList.remove(
              "is-over"
            );
          }
        );
      }
    );

    dz.addEventListener(
      "drop",
      function (e) {
        acceptFile(
          e.dataTransfer &&
          e.dataTransfer.files &&
          e.dataTransfer.files[0]
        );
      }
    );

    window.addEventListener(
      "dragover",
      function (e) {
        e.preventDefault();
      }
    );

    window.addEventListener(
      "drop",
      function (e) {
        e.preventDefault();
      }
    );

    window.addEventListener(
      "paste",
      function (e) {
        var items =
          e.clipboardData &&
          e.clipboardData.items;

        if (!items) {
          return;
        }

        for (
          var i = 0;
          i < items.length;
          i++
        ) {
          if (
            items[i]
              .type
              .indexOf(
                "image/"
              ) === 0
          ) {
            acceptFile(
              items[i]
                .getAsFile()
            );

            break;
          }
        }
      }
    );

    $("btn-clear")
      .addEventListener(
        "click",
        clearImage
      );

    $("btn-rerun")
      .addEventListener(
        "click",
        run
      );

    $("btn-sample")
      .addEventListener(
        "click",
        useSample
      );

    /* Scoped by attribute, not by class. Both groups are .seg-btn, so one
       shared handler reading dataset.k set state.k to NaN whenever a view
       button was clicked. */
    function wireSegGroup(selector, apply) {
      document.querySelectorAll(selector).forEach(function (b) {
        b.addEventListener("click", function () {
          document.querySelectorAll(selector).forEach(function (o) {
            o.classList.remove("is-on");
          });
          b.classList.add("is-on");
          apply(b);
          if (state.key) {
            run();
          } else {
            renderResults();
          }
        });
      });
    }

    wireSegGroup("[data-k]", function (b) {
      state.k = Number(b.dataset.k);
    });

    wireSegGroup("[data-view]", function (b) {
      state.view = b.dataset.view;
    });

    renderModelCard();
    renderAll();
    detectBackend();
  }

  if (
    document.readyState ===
    "loading"
  ) {
    document.addEventListener(
      "DOMContentLoaded",
      init
    );
  } else {
    init();
  }
})();
