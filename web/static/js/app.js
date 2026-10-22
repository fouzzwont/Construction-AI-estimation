/* Construction AI Estimation - front end controller */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const CUR = window.APP.currency || "₹";
  const LIMIT_MB = window.APP.uploadLimit || 80;

  const el = {
    dropzone: $("#dropzone"),
    fileInput: $("#fileInput"),
    dzIcon: $("#dzIcon"),
    dzFile: $("#dzFile"),
    dzFileName: $("#dzFileName"),
    dzFileSize: $("#dzFileSize"),
    analyseBtn: $("#analyseBtn"),
    analyseLabel: $("#analyseLabel"),
    projectName: $("#projectName"),
    scaleSelect: $("#scaleSelect"),
    marketSelect: $("#marketSelect"),
    progressWrap: $("#progressWrap"),
    progressStage: $("#progressStage"),
    progressPct: $("#progressPct"),
    progressBar: $("#progressBar"),
    steps: $("#steps"),
    errorBox: $("#errorBox"),
    resultsPanel: $("#resultsPanel"),
    resultSubtitle: $("#resultSubtitle"),
    tiles: $("#tiles"),
    warningBox: $("#warningBox"),
    boqBody: $("#boqTable tbody"),
    rebarBody: $("#rebarTable tbody"),
    rebarEmpty: $("#rebarEmpty"),
    totalsBox: $("#totalsBox"),
    pageTabs: $("#pageTabs"),
    pageImage: $("#pageImage"),
    legend: $("#legend"),
    assumptionList: $("#assumptionList"),
    pageNotes: $("#pageNotes"),
    csvBtn: $("#csvBtn"),
    pdfBtn: $("#pdfBtn"),
    resetBtn: $("#resetBtn"),
  };

  let selectedFile = null;
  let jobId = null;
  let pollTimer = null;
  let currentPage = 0;

  // ------------------------------------------------------------------ //
  // helpers
  // ------------------------------------------------------------------ //
  const fmtNum = (v, d = 2) =>
    Number(v || 0).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });

  const fmtMoney = (v) => CUR + Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });

  const fmtBytes = (b) => {
    if (!b) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(b) / Math.log(1024));
    return `${(b / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${units[i]}`;
  };

  function showError(msg) {
    el.errorBox.hidden = false;
    el.errorBox.textContent = msg;
    el.analyseBtn.disabled = !selectedFile;
  }

  function clearError() {
    el.errorBox.hidden = true;
    el.errorBox.textContent = "";
  }

  // ------------------------------------------------------------------ //
  // file selection
  // ------------------------------------------------------------------ //
  function setFile(file) {
    if (!file) return;
    if (!/\.pdf$/i.test(file.name)) {
      showError("That file is not a PDF. Please upload a drawing in PDF format.");
      return;
    }
    selectedFile = file;
    clearError();
    el.dzFile.hidden = false;
    el.dzFileName.textContent = file.name;
    el.dzFileSize.textContent = fmtBytes(file.size);
    el.dropzone.classList.add("has-file");
    el.dzIcon.style.display = "none";
    el.dropzone.querySelector(".dz-title").textContent = "Drawing ready";
    el.dropzone.querySelector(".dz-sub").innerHTML = "Click to choose a different file";
    el.analyseBtn.disabled = false;
  }

  // Some browsers fire a synthetic click on the dropzone after a drop. Without
  // this guard the file picker would reopen over the freshly dropped file.
  let droppedRecently = false;

  el.dropzone.addEventListener("click", () => {
    if (droppedRecently) return;
    el.fileInput.click();
  });
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });
  el.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove("drag"); })
  );
  el.dropzone.addEventListener("drop", (e) => {
    droppedRecently = true;
    setTimeout(() => { droppedRecently = false; }, 0);
    setFile(e.dataTransfer.files[0]);
  });

  el.scaleSelect.addEventListener("change", () => {
    const v = el.scaleSelect.value;
    $("#scaleHint").textContent = v
      ? `Scale forced to 1:${v}. Use this when the title block is missing.`
      : "Leave on auto — the engine reads the title block and dimension lines.";
  });

  // ------------------------------------------------------------------ //
  // analysis
  // ------------------------------------------------------------------ //
  el.analyseBtn.addEventListener("click", startAnalysis);

  async function startAnalysis() {
    if (!selectedFile) return;
    clearError();
    el.analyseBtn.disabled = true;
    el.analyseLabel.textContent = "Analysing…";
    el.progressWrap.hidden = false;
    el.resultsPanel.hidden = true;
    setProgress(0, "Uploading drawing…");
    setSteps(-1);

    const fd = new FormData();
    fd.append("pdf", selectedFile);
    if (el.scaleSelect.value) fd.append("scale", el.scaleSelect.value);
    if (el.marketSelect.value) fd.append("market", el.marketSelect.value);
    if (el.projectName.value.trim()) fd.append("project_name", el.projectName.value.trim());

    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Upload failed.");
      jobId = data.job_id;
      poll();
    } catch (err) {
      showError(err.message || "Something went wrong during upload.");
      el.analyseBtn.disabled = false;
      el.analyseLabel.textContent = "Analyse drawing";
      el.progressWrap.hidden = true;
    }
  }

  function poll() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/job/${jobId}`);
        const data = await res.json();
        if (data.status === "done") {
          setProgress(1, "Done");
          setSteps(99);
          await loadResult();
          return;
        }
        if (data.status === "error") {
          showError(data.error || "Analysis failed.");
          el.analyseBtn.disabled = false;
          el.analyseLabel.textContent = "Analyse drawing";
          return;
        }
        setProgress(data.progress || 0, data.stage || "Working…");
        updateStep(data.stage);
        poll();
      } catch (err) {
        showError("Lost connection to the analysis server.");
        el.analyseBtn.disabled = false;
        el.analyseLabel.textContent = "Analyse drawing";
      }
    }, 600);
  }

  function setProgress(pct, stage) {
    el.progressBar.style.width = `${Math.round(pct * 100)}%`;
    el.progressPct.textContent = `${Math.round(pct * 100)}%`;
    el.progressStage.textContent = stage;
  }

  function setSteps(doneIndex) {
    $$("#steps li").forEach((li, i) => {
      li.classList.toggle("done", i < doneIndex);
      li.classList.toggle("active", i === doneIndex);
    });
  }

  function updateStep(stage) {
    const map = {
      "Parsing drawing": 0,
      "Detecting scale and elements": 1,
      "Estimating quantities": 3,
      "Annotating drawings": 3,
      "Building reports": 4,
    };
    const idx = map[stage];
    if (idx === undefined) return;
    $$("#steps li").forEach((li, i) => {
      li.classList.toggle("active", i === idx);
      li.classList.toggle("done", i < idx);
    });
  }

  // ------------------------------------------------------------------ //
  // results
  // ------------------------------------------------------------------ //
  async function loadResult() {
    const res = await fetch(`/api/result/${jobId}`);
    const data = await res.json();
    if (!data.result) return;

    render({ ...data.result, elapsed_seconds: data.elapsed_seconds }, data.pages);
    el.resultsPanel.hidden = false;
    el.analyseBtn.disabled = false;
    el.analyseLabel.textContent = "Analyse drawing";
    el.resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function render(result, pages) {
    const q = result.quantities || {};
    el.resultSubtitle.textContent =
      `${result.project_name} · ${result.source_file} · ${result.pages.length} page(s) analysed in ${Number(result.elapsed_seconds || 0).toFixed(1)}s`;

    // ---- tiles ------------------------------------------------------- //
    const conf = Math.round((result.confidence || 0) * 100);
    const confClass = conf >= 70 ? "ok" : conf >= 45 ? "warn" : "err";
    const tiles = [
      { k: "Total estimate", v: fmtMoney(result.grand_total), s: `${result.market} rates`, hero: true },
      { k: "Drawing scale", v: (result.scale && result.scale.ratio) || "n/a", s: `via ${(result.scale && result.scale.method) || "n/a"}` },
      { k: "Storeys", v: result.storeys, s: "detected from notes" },
      { k: "Concrete", v: `${fmtNum(q.concrete_m3)} m³`, s: `${result.boq.filter((l) => l.category === "Concrete").length} item(s)` },
      { k: "Steel", v: `${fmtNum(q.steel_t, 3)} t`, s: `${fmtNum(q.steel_kg, 0)} kg` },
      { k: "Formwork", v: `${fmtNum(q.formwork_m2)} m²`, s: "shuttering" },
      { k: "Elements", v: q.elements || 0, s: "detected" },
      { k: "Confidence", v: `${conf}%`, s: "estimate reliability", cls: confClass },
    ];
    el.tiles.innerHTML = tiles
      .map(
        (t) => `<div class="tile${t.hero ? " hero" : ""}">
            <div class="k">${t.k}</div>
            <div class="v ${t.cls || ""}">${t.v}</div>
            <div class="s">${t.s}</div>
          </div>`
      )
      .join("");

    // ---- warnings ----------------------------------------------------- //
    el.warningBox.innerHTML = (result.warnings || [])
      .map((w) => `<div class="alert alert-warn">${w}</div>`)
      .join("");

    // ---- BOQ ---------------------------------------------------------- //
    el.boqBody.innerHTML = (result.boq || [])
      .map(
        (l) => `<tr>
            <td><span class="cat-pill">${l.category}</span></td>
            <td>${l.item}${l.detail ? `<br><span class="muted" style="font-size:11.5px">${l.detail}</span>` : ""}</td>
            <td class="num">${fmtNum(l.quantity, l.unit === "m3" ? 2 : 2)}</td>
            <td class="muted">${l.unit}</td>
            <td class="num">${fmtMoney(l.rate)}</td>
            <td class="num"><strong>${fmtMoney(l.amount)}</strong></td>
          </tr>`
      )
      .join("") || `<tr><td colspan="6" class="muted">No priced items were derived from this drawing.</td></tr>`;

    // ---- totals -------------------------------------------------------- //
    const totals = result.totals || {};
    el.totalsBox.innerHTML =
      Object.entries(totals)
        .map(([cat, amt]) => `<div class="row"><span>${cat}</span><span class="num">${fmtMoney(amt)}</span></div>`)
        .join("") +
      `<div class="row grand"><span>Total estimate</span><span class="num">${fmtMoney(result.grand_total)}</span></div>`;

    // ---- rebar --------------------------------------------------------- //
    const rebar = result.rebar || [];
    el.rebarEmpty.hidden = rebar.length > 0;
    el.rebarBody.innerHTML = rebar
      .map(
        (r) => `<tr>
            <td><strong>Ø${r.diameter_mm}</strong></td>
            <td class="num">${fmtNum(r.total_length_m, 1)}</td>
            <td class="num">${r.count}</td>
            <td class="num">${fmtNum(r.weight_kg, 1)}</td>
            <td class="num">${fmtNum(r.weight_t, 3)}</td>
          </tr>`
      )
      .join("");

    // ---- drawings ------------------------------------------------------ //
    el.pageTabs.innerHTML = (pages || [])
      .map((p, i) => `<button class="vtab${i === 0 ? " active" : ""}" data-page="${p.index}">Page ${p.index + 1}</button>`)
      .join("");
    $$(".vtab").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".vtab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentPage = parseInt(btn.dataset.page, 10);
        el.pageImage.src = `/api/page/${jobId}/${currentPage}`;
      })
    );
    if (pages && pages.length) {
      currentPage = pages[0].index;
      el.pageImage.src = `/api/page/${jobId}/${currentPage}`;
    }

    const counts = {};
    (pages || []).forEach((p) => {
      Object.entries(p.counts || {}).forEach(([k, v]) => (counts[k] = (counts[k] || 0) + v));
    });
    const palette = { rebar: "#1e1ede", column: "#d62728", beam: "#2ca02c", slab: "#ffa500", wall: "#db27dc", footing: "#008cff" };
    el.legend.innerHTML = Object.entries(counts)
      .map(([k, v]) => `<span><i style="background:${palette[k] || "#5b8cff"}"></i>${k}: ${v}</span>`)
      .join("");

    // ---- assumptions ---------------------------------------------------- //
    el.assumptionList.innerHTML = (result.assumptions || []).map((a) => `<li>${a}</li>`).join("");
    const notes = [];
    (pages || []).forEach((p) => (p.notes || []).forEach((n) => notes.push(`<strong>Page ${p.index + 1}:</strong> ${n}`)));
    el.pageNotes.innerHTML = notes.join("") || "<li class='muted'>No notes.</li>";

    // ---- downloads ------------------------------------------------------ //
    el.csvBtn.hidden = false;
    el.pdfBtn.hidden = false;
    el.csvBtn.onclick = () => (window.location.href = `/api/download/${jobId}/csv`);
    el.pdfBtn.onclick = () => (window.location.href = `/api/download/${jobId}/pdf`);
  }

  // ------------------------------------------------------------------ //
  // tabs
  // ------------------------------------------------------------------ //
  $$(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      $$(".tab").forEach((t) => t.classList.remove("active"));
      $$(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`#tab-${tab.dataset.tab}`).classList.add("active");
    })
  );

  el.resetBtn.addEventListener("click", () => {
    selectedFile = null;
    el.fileInput.value = "";
    el.dzFile.hidden = true;
    el.dropzone.classList.remove("has-file");
    el.dzIcon.style.display = "";
    el.dropzone.querySelector(".dz-title").textContent = "Drag & drop a PDF blueprint";
    el.dropzone.querySelector(".dz-sub").innerHTML =
      'or <span class="link">browse your files</span> &middot; max ' + LIMIT_MB + " MB";
    el.progressWrap.hidden = true;
    el.resultsPanel.hidden = true;
    el.analyseBtn.disabled = true;
    el.analyseLabel.textContent = "Analyse drawing";
    clearError();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
