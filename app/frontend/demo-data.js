/* ==========================================================================
   demo-data.js — label vocabularies, published metrics, and the offline
   stand-in used when no model server is reachable.

   Every metric below is copied from the committed run artifacts; the source
   file for each block is named in a comment so the figures stay auditable.
   When the backend is live, /api/meta overrides FI.meta and FI.metrics.
   ========================================================================== */
window.FI = window.FI || {};

/* ------------------------------------------------- label vocabularies ---
   artifacts/task1/label_classes.json, artifacts/task2/task2_season_class_mapping.json,
   artifacts/task3/label_classes.json                                        */
FI.meta = {
  itemType: [
    "Accessory Gift Set", "Backpacks", "Bangle", "Basketballs", "Belts", "Booties",
    "Boxers", "Bra", "Bracelet", "Briefs", "Camisoles", "Capris", "Caps",
    "Casual Shoes", "Churidar", "Clutches", "Cufflinks", "Deodorant", "Dresses",
    "Duffel Bag", "Dupatta", "Earrings", "Flats", "Flip Flops", "Formal Shoes",
    "Foundation and Primer", "Fragrance Gift Set", "Free Gifts", "Gloves",
    "Handbags", "Heels", "Innerwear Vests", "Jackets", "Jeans", "Jeggings",
    "Jewellery Set", "Jumpsuit", "Kajal and Eyeliner", "Kurta Sets", "Kurtas",
    "Kurtis", "Laptop Bag", "Leggings", "Lip Liner", "Lipstick", "Lounge Pants",
    "Lounge Shorts", "Messenger Bag", "Mobile Pouch", "Mufflers", "Nail Polish",
    "Necklace and Chains", "Night suits", "Nightdress", "Patiala", "Pendant",
    "Perfume and Body Mist", "Ring", "Rompers", "Rucksacks", "Salwar", "Sandals",
    "Sarees", "Scarves", "Shirts", "Shoe Accessories", "Shorts", "Skirts",
    "Socks", "Sports Sandals", "Sports Shoes", "Stockings", "Stoles",
    "Sunglasses", "Suspenders", "Sweaters", "Sweatshirts", "Swimwear", "Ties",
    "Tops", "Track Pants", "Tracksuits", "Travel Accessory", "Trousers", "Trunk",
    "Tshirts", "Tunics", "Waist Pouch", "Waistcoat", "Wallets", "Watches",
    "Water Bottle"
  ],
  season: ["Fall", "Spring", "Summer", "Winter"],
  gender: ["Boys", "Girls", "Men", "Unisex", "Women"],
  usage:  ["Casual", "Ethnic", "Formal", "Sports"]
};

/* --------------------------------------------------------- model card ---
   task1_summary.json · task2_season_metrics.json · task3_summary.json ·
   task4_summary.json + search_manifest.json                                */
FI.metrics = {
  source: "artifacts/task{1,2,3,4}/*.json",
  tasks: [
    {
      id: "Task 1",
      name: "Item type",
      headline: 20.8,
      headlineLabel: "top-1 accuracy",
      detail: "Top-3 <b>37.2%</b> · top-5 <b>47.7%</b> · weighted F1 <b>15.1</b> over 44 classes. " +
              "Majority-class baseline is <b>18.6%</b>.",
      flag: "warn",
      flagText: "Barely above baseline — read the alternatives, not the top-1",
      note: "A classical feature pipeline reached 56.4 weighted F1 on the same split, so the " +
            "deployed CNN is the weaker of the two options investigated."
    },
    {
      id: "Task 2",
      name: "Season",
      headline: 67.5,
      headlineLabel: "top-1 accuracy",
      detail: "Macro F1 <b>64.3</b> · balanced accuracy <b>69.9</b> across 4 seasons. " +
              "Majority-class baseline is <b>49.6%</b>.",
      flag: "warn",
      flagText: "Usable signal, frequent confusion between adjacent seasons",
      note: "Season is weakly determined by appearance alone; Fall/Spring items are often visually identical."
    },
    {
      id: "Task 3",
      name: "Gender &amp; occasion",
      headline: 89.3,
      headlineLabel: "gender accuracy",
      detail: "Occasion accuracy <b>91.1%</b> · exact match on both heads <b>81.3%</b> · " +
              "cross-validated macro F1 <b>80.5 ± 1.1</b>.",
      flag: "good",
      flagText: "Strong and stable across folds",
      note: "Multi-task CNN pruned to 80% sparsity with no measurable loss, 2.7M parameters."
    },
    {
      id: "Task 4",
      name: "Visual search",
      headline: 81.2,
      headlineLabel: "precision@10",
      detail: "mAP@10 <b>76.7</b> · nDCG@10 <b>89.6</b> · R-precision <b>80.3</b> on 2,000 held-out " +
              "queries against a 32,837-item catalogue. Random baseline is <b>5.9%</b>.",
      flag: "good",
      flagText: "0.01 ms per query — comfortably real-time",
      note: "Same-colour agreement is lower at 56.3%; the embedding favours silhouette over colour."
    }
  ]
};

/* ------------------------------------------------------ sample queries ---
   Ids drawn from the held-out test split so a live backend can serve the
   real thumbnail for them.                                                 */
FI.samples = [
  { id: 52003, label: "Sample · 52003" },
  { id: 52017, label: "Sample · 52017" },
  { id: 52044, label: "Sample · 52044" },
  { id: 52070, label: "Sample · 52070" },
  { id: 52101, label: "Sample · 52101" }
];

/* ------------------------------------------------------------- colours ---
   Used to tint result tiles by predicted base colour.                      */
FI.colourHex = {
  Black: "#2B2A28", White: "#F2F0EC", Grey: "#9A968E", Blue: "#3E6BA8",
  "Navy Blue": "#2A3A5E", Red: "#B4413C", Pink: "#D98BA5", Green: "#5A8A5E",
  Purple: "#7B6096", Brown: "#8A6A4E", Beige: "#D8C7AC", Yellow: "#D9B23F",
  Orange: "#D08040", Maroon: "#7A3540", Silver: "#C3C0BA", Gold: "#C6A45C",
  Olive: "#7B7A44", Teal: "#3F8188", Cream: "#EBE0CB", Khaki: "#B5A57C"
};

/* ==========================================================================
   Offline stand-in
   --------------------------------------------------------------------------
   Deterministic: the same image always yields the same numbers, so a demo
   can be re-run without the figures dancing around. This is a presentation
   placeholder — it does NOT run the models.
   ========================================================================== */
FI.demo = (function () {

  /* mulberry32 — small seeded PRNG */
  function rng(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6D2B79F5) >>> 0;
      let x = Math.imul(t ^ (t >>> 15), 1 | t);
      x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function seedFrom(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  /* Draw a descending probability vector, normalised, with the winner's
     share controlled by `peak`. Mirrors how a softmax actually looks. */
  function distribution(classes, rand, peak, k) {
    const picked = [];
    const pool = classes.slice();
    for (let i = 0; i < Math.min(k, pool.length); i++) {
      picked.push(pool.splice(Math.floor(rand() * pool.length), 1)[0]);
    }
    let remaining = 1 - peak;
    const probs = [peak];
    for (let i = 1; i < picked.length; i++) {
      const share = i === picked.length - 1 ? remaining : remaining * (0.45 + rand() * 0.25);
      probs.push(share);
      remaining -= share;
    }
    return picked.map((label, i) => ({ label: label, p: Math.max(probs[i], 0.001) }));
  }

  const COLOURS = Object.keys(FI.colourHex);

  function predict(key) {
    const rand = rng(seedFrom(key));

    /* Peaks chosen to reflect each task's published reliability: item type
       is genuinely low-confidence, gender and occasion are not. */
    const itemType = distribution(FI.meta.itemType, rand, 0.11 + rand() * 0.22, 5);
    const season   = distribution(FI.meta.season,   rand, 0.42 + rand() * 0.30, 4);
    const gender   = distribution(FI.meta.gender,   rand, 0.74 + rand() * 0.22, 4);
    const usage    = distribution(FI.meta.usage,    rand, 0.71 + rand() * 0.25, 4);

    return {
      demo: true,
      latency_ms: Math.round(28 + rand() * 34),
      predictions: {
        item_type: itemType,
        season:    season,
        gender:    gender,
        usage:     usage
      }
    };
  }

  function search(key, k) {
    const rand = rng(seedFrom(key + "|search"));
    const types = FI.meta.itemType;
    /* Neighbours cluster around a small family of article types, the way a
       real embedding index returns them, rather than spreading uniformly. */
    const family = [
      types[Math.floor(rand() * types.length)],
      types[Math.floor(rand() * types.length)],
      types[Math.floor(rand() * types.length)]
    ];

    let sim = 0.93 + rand() * 0.06;
    const out = [];
    for (let i = 0; i < k; i++) {
      const r = rand();   /* 55% primary, 27% secondary, 18% tertiary */
      out.push({
        id: 10000 + Math.floor(rand() * 42000),
        similarity: Math.max(sim, 0.4),
        article_type: family[r < 0.55 ? 0 : r < 0.82 ? 1 : 2],
        base_colour: COLOURS[Math.floor(rand() * COLOURS.length)]
      });
      sim -= 0.006 + rand() * 0.018;
    }
    return { demo: true, latency_ms: Math.round(1 + rand() * 3), results: out };
  }

  return { predict: predict, search: search };
})();
