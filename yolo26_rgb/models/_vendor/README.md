# Vendored Ultralytics source

Any code copied or adapted from Ultralytics' YOLO26 source goes here, nowhere else in this repo.

Rules for anything added to this directory:

1. Keep the original Ultralytics copyright notice at the top of every file.
2. Add an AGPL-3.0 header (this whole repo is AGPL-3.0, but be explicit in files that are directly derived from Ultralytics' code, not just adapted).
3. Note in a comment which upstream file/commit/version it was copied from, for traceability.
4. Nothing outside `yolo26_rgb/models/` should import from this directory directly, go through `yolo26_rgb/models/heads.py` or a similar wrapper so the vendored boundary stays clear.

Empty for now, scaffolding only.
