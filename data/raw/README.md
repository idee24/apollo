# data/raw/ — obtain sources yourself (nothing here is committed)

Raw data files are **gitignored** and must never be committed or redistributed. See
licence terms per source in [`../registry.yaml`](../registry.yaml) and `STRATEGY.md` §5.2.

## GTD (required for Phase 1)
1. Go to the GTD download page: https://www.start.umd.edu/download-global-terrorism-database
2. Complete the request form and accept the Terms of Use (non-commercial research only).
3. Place the file(s) here, e.g. `data/raw/gtd/globalterrorismdb_<version>.csv`.
4. Record the exact version, `local_path`, and `sha256` in `../registry.yaml`.
   - Compute the hash (Git Bash): `sha256sum data/raw/gtd/<file>.csv`
   - Or PowerShell: `Get-FileHash data\raw\gtd\<file>.csv -Algorithm SHA256`

Do **not** expose raw GTD records or free-text narratives through any client/API —
serve only derived model outputs.

## Other sources
Add each only when its Phase unlocks (UCDP/WDI/V-Dem in the Model B research track).
ACLED is **blocked for training**; do not place ACLED data here for modelling use.
