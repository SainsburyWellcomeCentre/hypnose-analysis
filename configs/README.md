# configs

User-facing setup configuration (e.g. rig/olfactometer `.yml` used when running
experiments). Populated during the restructuring.

Note: the harp **device schemas** (`behavior.yml`, `olfactometer.yml`) consumed
by the analysis code remain *package data* under `src/hypnose/resources/device_schemas/`,
loaded via `importlib.resources` — they are not duplicated here.
