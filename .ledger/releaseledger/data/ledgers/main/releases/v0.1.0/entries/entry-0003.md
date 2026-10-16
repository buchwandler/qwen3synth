---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0003
release_version: v0.1.0
kind: removed
summary:
  Removed the qwen3synth CLI; use the Python API for synthesis and OnnxVoice
  for model management
status: accepted
audience: null
scopes: []
source_refs:
  - git:b5e4621ec9b9d5d221745202fc31cbb04de11127
paths:
  - qwen3synth/__main__.py
  - examples/00_discover_bundles.py
  - examples/01_voice_design_prompt_to_wav.py
  - examples/02_prepare_voice_profile.py
  - examples/03_synthesize_cloned_voice.py
  - examples/04_local_bundle.py
  - examples/05_inspect_voice_profile.py
  - tests/test_api_only.py
issues: []
prs: []
sources:
  - git:b5e4621ec9b9d5d221745202fc31cbb04de11127
contributors:
  - "@holgern"
breaking: false
internal: false
order: 3
---
