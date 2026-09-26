# Bundled fonts

Self-hosted so the viewer renders as designed without internet access (the packaged zip, locked-down networks).
Latin subsets, woff2, taken unmodified from the Fontsource npm packages (v5.3.0):

| Family | Weights | Licence |
|---|---|---|
| Barlow Condensed | 500, 600 | SIL Open Font License 1.1 — `OFL-BarlowCondensed.txt` |
| IBM Plex Sans | 400, 500, 600 | SIL Open Font License 1.1 — `OFL-IBMPlexSans.txt` |
| IBM Plex Mono | 400, 500 | SIL Open Font License 1.1 — `OFL-IBMPlexMono.txt` |

Declared with `@font-face` at the top of `src/styles.css`; Vite fingerprints the files into `dist/assets/`.
Glyphs outside the Latin subset (arrows such as → and ↗) fall back to the system font per character.
