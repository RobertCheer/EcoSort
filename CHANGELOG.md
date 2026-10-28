# Changelog

All notable changes to the EcoSort system. The format follows Keep a Changelog.

## [1.0.0] — 2026-11-07 — Final project

### Added
- Seven integrated sub-systems: capture, pre-processing, classifier, routing,
  actuation, logging and dashboard, joined by `ecosort/pipeline.py`.
- Streamlit interface with five views: sort an item, live dashboard, review queue,
  batch evaluation and about.
- Human review queue. An item below the threshold arrives with its photo, the
  model's guess, the runner-up and an inspection hint. A person decides the class,
  and that decision is what opens a bin.
- Animated, labelled bin board. The chosen lid opens and the item drops in.
- Per-backend confidence threshold. The CNN uses 0.60, the SVM uses 0.35.
- Frame trimming, so a screenshot border cannot change a verdict.
- 88 automated tests.
- `reproduce_report.py` re-checks all ten recorded numbers in under a second.
- `stress_test.py` measures accuracy outside the studio benchmark.

### Changed
- Best model moved from MobileNetV2 at 128 px (0.776) to a MobileNetV3 blend at
  224 px (0.921).
- Augmentation switched to random resized crop, scale 0.6 to 1.0.
- Optimiser switched to AdamW with weight decay 0.02 and a cosine schedule.
- The epoch budget is now fixed at 12, and the final epoch ships, so the validation
  set no longer selects the model.
- The classifier is cached separately from the threshold, so the interface no
  longer reloads 60 MB of models on every slider move.

### Fixed
- Held-out items lost their ground truth when passed as bytes.
- `item_0001` was reused across runs, so one review photo overwrote another.
- The review queue lost every photo after a restart. Each log row now stores the
  threshold that applied and a reference to the source photo.
- The review text quoted the current threshold instead of the applied one, which
  produced self-contradictory sentences such as "78% is below the 60% threshold".

### Known limitations
- TrashNet is studio photography on white backgrounds. Accuracy falls to 0.36 when
  the object is small in a cluttered frame.
- Clear, colourless glass is under-represented. The station sends such an item to a
  person rather than guessing.
- Actuation is simulated, as the unit brief allows for a prototype.
