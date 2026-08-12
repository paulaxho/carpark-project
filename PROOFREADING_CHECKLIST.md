# Final Proofreading Checklist — car_park dissertation

Work through top to bottom. Locations are by section/table/figure. Tick each box as you fix it.
Reference titles and code tokens are exempt from the spelling pass (see §5).

---

## 1. Raw placeholder text still printing in the PDF (do these first)

- [ ] **Table 4 (Software environment, p.35) — NVIDIA driver row.** Currently prints
      `<RTX 4070 Ti SUPER box>; <RTX 3090 Ti box>`. Replace with the two real driver
      versions, or `not recorded (GPU server no longer accessible)`.
- [ ] **References — Jocher et al. (Ultralytics).** Ends "Software; set exact version from
      training logs." → change note to `Software, version 8.2.0.`
- [ ] **References — Getmapping Plc.** Ends "Downloaded via UCL licence; use the exact
      tiles/date from the Digimap-generated citation." → replace instruction with the real note.
- [ ] **References — GEOLYTIX.** Ends "Insert the exact version (e.g. v31) and access date
      used for site selection." → insert the actual version + access date.
- [ ] **References — CVAT.ai Corporation.** Ends "Commit <hash>." → insert the real commit hash.
- [ ] **References — Shreeram H et al. (2025).** Ends "p. TODO: from IEEE Xplore." → insert
      the page range / DOI, or remove.

## 2. Broken cross-reference

- [ ] **§3.13 body:** "provided in **Appendix 6.3**" → **Appendix D** (hardcode `Appendix~D`).
- [ ] **Table 4 caption:** "in **Appendix 6.3**" → **Appendix D**.

## 3. Typos and broken sentences

- [ ] **§6.3:** "the empirical **analysisd** confirmed that **the** 14 px is the optimal footprint"
      → "the empirical **analysis** confirmed that 14 px is the optimal footprint."
- [ ] **§3.9:** "…the most generous fair treatment**.** The baseline was evaluated…"
      → change the full stop to a comma: "…the most generous fair treatment**,** the baseline
      was evaluated…". (Also consider "the fairest possible treatment".)
- [ ] **§3.9:** "so that any detection overlapping a true vehicle**,** counts as a match"
      → delete the comma after "vehicle".
- [ ] **§3.12:** "…a measured, per-site statistic because every detection is georeferenced**, .**"
      → "…a measured, per-site statistic**, because every detection is georeferenced.**"

## 4. Results-section grammar (corrected earlier — not present in this PDF)

- [ ] **§4.1:** "a **well documented** pattern" → "well-documented".
- [ ] **§4.1:** "…for COWC's parking lots**. Whereas** the denser source…"
      → "…parking lots**, whereas** the denser source…"
- [ ] **§4.1:** "…to 0.696**. A gain** of 0.49…" → "…to 0.696**, a gain** of 0.49…"
- [ ] **§4.3:** "on VEDAI it **scored** 0.696" → "on VEDAI**, 0.696**" (tense).
- [ ] **§4.3:** "…from 0.528 to 0.886**. A recovery** of 0.36…"
      → "…from 0.528 to 0.886**, a recovery** of 0.36…"
- [ ] **§4.3:** "…inexpensive to close**,** a calibration set of eight sites…"
      → "…inexpensive to close**. A** calibration set…" (comma splice).
- [ ] **§4.4:** "…finds and localises vehicles**, however** the operational question is…"
      → "…finds and localises vehicles**. The operational question, however, is**…"
- [ ] **§4.4:** "The two biggest residuals**:** UK028 (…) and UK013 (…), fall at opposite ends"
      → replace the colon with a comma: "The two biggest residuals**,** UK028 …"

## 5. UK/US spelling (document is UK English)

Change each to the UK form. Do NOT change these inside published reference titles
(e.g. Lipton "maximize F1", Lu "characterizing") or code tokens (`amp`, etc.).

- [ ] §2.6: "generali**z**ation" (×2) → generalisation
- [ ] §2.6: "characteri**z**ed" → characterised
- [ ] §2.6: "**labeled**" → labelled
- [ ] §2.6: "maximi**z**e" → maximise
- [ ] Fig. 7 caption: "maximi**z**e vehicle coverage" → maximise
- [ ] §2.5: "object **centers**" (×3) → centres
- [ ] §2.5: "populari**z**ed" → popularised
- [ ] §2.5: "recogni**z**ed" → recognised
- [ ] §2.5: "**favors**" → favours
- [ ] §3.1: "initiali**z**ed" → initialised
- [ ] §3.1 list: "Standardi**z**ation" → Standardisation
- [ ] Fig. 8 caption: "**Centering** of a fixed…" → Centring
- [ ] Fig. 11 caption: "categori**z**ed by fold" → categorised
- [ ] Table 7 caption: "calibration-optimi**z**ed" → calibration-optimised
- [ ] §4.5: "the locali**z**ed detection centroids" → localised
- [ ] §4.5: "summari**z**ed in Table 8" → summarised
- [ ] §6.3: "improve the strict-IoU locali**z**ation" → localisation

## 6. Section 2.5 — three fragments + quote style

- [ ] "…relatively shallow feature maps**. A design** its own authors identify as…"
      → "…shallow feature maps**, a design** its own authors identify as…"
- [ ] "…to separate classification from box regression**. An arrangement** popularised…"
      → "…from box regression**, an arrangement** popularised…"
- [ ] "…the fix is architectural**. Specifically, incorporating**…"
      → "…the fix is architectural**—specifically, incorporating**…"
- [ ] §2.6: straight quotes `"domain gap"` → curly quotes `` ``domain gap'' ``

## 7. Minor consistency

- [ ] Fig. 11 caption: "detailed in **table 3**" → "Table 3" (capitalise).
- [ ] §4.5: "counting and occupancy **estimations reduce**" → "the counting and occupancy
      **estimates reduce**".
- [ ] **Intro vs Conclusion scope mismatch.** Conclusion says "The barrier to **estimating UK
      car-park occupancy**…" (with the premise-vs-claim demarcation); the Introduction still
      says "The barrier to **measuring UK retail activity**…" with no demarcation. Apply the
      intro central-argument revision so both chapters state the scope identically.

---

## Not errors — no action needed
PDF text-extraction merged some hyphenated line-breaks ("Highresolution", "axisaligned",
"finetuning", "groundtruth", etc.). These render correctly hyphenated in the actual
document. Ignore them.

---

## Final step before submission
- [ ] Recompile (pdflatex → bibtex → pdflatex → pdflatex) and re-scan `main.log` for
      "undefined" citations/references and any remaining `<...>` or `TODO` strings.
