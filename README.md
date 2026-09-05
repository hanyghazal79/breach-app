---
title: BREACH Score Web App
emoji: 🎗️
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: "1.63.0"
app_file: src/app.py
pinned: false
license: mit
---

# BREACH Score Web App

Research tool for computing the BREACH score alongside NPI and CTS5,
and comparing their discrimination performance on an uploaded cohort.

**This is a research tool, not for clinical decision-making.**
BREACH is an experimental, unvalidated score.

Upload de-identified data only. Cohort files are processed in memory
and are never saved to disk — data is discarded when your session ends.