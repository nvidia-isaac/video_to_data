# Workflow GIFs

- Source: release `0979067b` plus viewer-reset fix `e2048669`.
- Renderer: Viser; camera framing and ground-grid visibility adjusted for capture.
- Debug overlays hidden with `markers.enabled=false` in replay/policy captures; use `markers.enabled=true` to show them.
- Robot/scene GIFs: camera orbit around a static pose.
- Section 3 GIFs: one complete kinematic reference playback via **Show kinematic reference**, without physics.
- Policy GIFs: deterministic rollouts from frame 0 using the terminal EVT-07 actors; Sharpa ends after placement, before retreat.
- Playback: 10 fps at the captured playback speed. GIF looping restarts the recording.

| Actor | Training steps | SHA-256 |
| --- | ---: | --- |
| Sharpa tissue-box | 250,003,456 | `8d27a10de356e40ff4912439d6ddf5ecb265f567fbee438b481c73a7549ba04c` |
| G1 snack-box | 349,999,104 | `37d0af55b8310bbdaa55176b8defe63d486eff40d2c28fa041aa0ecee5ae815f` |

[Capture metadata and GIF hashes](manifest.json).
