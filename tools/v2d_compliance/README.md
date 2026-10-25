# V2D Challenge Submission Compliance & Anti-Reward-Hacking CI/CD

This package provides a participant-facing CI/CD pipeline, offline compliance
auditor, rolling weekly quota ledger, and autonomous-agent guardrail for the
[NVIDIA Video-to-Data (V2D) Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

## Why this CI/CD exists (the autonomous / RSI model trap)

The V2D Challenge submission kit includes an agent skill (`skills/v2d-submit/SKILL.md`)
so participants can use coding agents to package and validate entries. However,
when participants run autonomous coding agents or Recursive Self-Improvement (RSI)
harnesses (Claude Code, Codex, Cursor, OpenHands, SWE-agent, etc.), four failure
modes routinely cause rule violations and team suspensions:

1. **Kaggle's 5/day UI limit vs. the stated 5/week challenge rule:**
   Kaggle's competition settings only support a daily submission cap (`5/day`),
   whereas the official V2D Challenge rules state: *"five (5) Submissions per
   week, with unlimited Submissions during the final three (3) days of the
   Competition Period"*. An autonomous agent checking `limits.num_allowed_now`
   sees `5 left today` and uploads up to 35 times per week per leaderboard.
2. **Autonomous live-upload probing ("testing whether the submission can be sent"):**
   Agents frequently execute `submit/v2d_submit.py` without `--dry-run` (or invoke
   `kaggle competitions submit` directly) just to test end-to-end command execution,
   burning weekly quota and uploading unverified artifacts.
3. **Cross-track sprawl & per-leaderboard split submissions (`--only` probing):**
   V2D uses 23 separate Kaggle competitions across 5 logical track/tier groups
   (`track_1` [5], `track_2_tier1` [4], `track_2_tier2` [4], `track_2_tier3` [4],
   `track_3` [6]). Agents optimizing individual metrics may pass `--only` to upload
   different tuned files to individual metric leaderboards within the same track,
   upload standalone meshes to `track_3_mesh`, or upload to tracks/tiers the team
   did not enter.
4. **Local reward-hacking on provenance, meshes, and cross-tier data:**
   Agents optimizing against local checks may fabricate `code_commit_url` or
   `checkpoint_sha256` digests, add `1e-7 m` Gaussian noise to meshes to evade
   exact byte-SHA duplicate checks, copy Track 2 Tier 1 clean ground-truth poses
   into Tier 2/3, or freeze Track 1 rotations to game `PEN = 0`.

---

## Quickstart for participant repositories

Every V2D participant repository (shared with `@Nvidia-V2D-Challenge` under Rule
2.9.a) can enable these protections in three steps:

### 1. Initialize your team's track & quota lock (`.v2d_policy.json`)

Lock the specific tracks/tiers your team is entering so no script or agent can
upload outside your declared scope:

```bash
python3 tools/compliance_gate.py init-policy \
  --tracks track_1 track_2_tier1 track_3 \
  -o .v2d_policy.json
```

### 2. Install the GitHub Actions CI/CD workflow

Copy `participant_ci_template.yml` (also bundled inside `v2d_submission_kit.zip`
at `.github/workflows/v2d_submission_ci.yml`) into your submission repository:

```bash
mkdir -p .github/workflows
cp tools/v2d_compliance/participant_ci_template.yml .github/workflows/v2d_submission_ci.yml
```

- **On every `push` and `pull_request` (CI — 100% offline, zero Kaggle uploads):**
  - Scans all `.py` and `.sh` scripts with `agent_submission_guard.py --scan-scripts .`
    to ensure no script embeds direct Kaggle upload calls.
  - Audits staged submission files with `compliance_gate.py audit` for schema,
    commit provenance, checkpoint file SHA-256 parity, `< 0.1 mm` (`1e-4 m`)
    microscopic mesh-noise evasion, `< 1.0 mm` reference scan copying, cross-tier
    trajectory leakage, and Track 1 kinematic non-degeneracy.
- **On manual `workflow_dispatch` (CD — human-gated release):**
  - `mode=dry-run` authenticates with Kaggle and verifies rules acceptance, team
    identity parity, and the **rolling 7-day (`5/week`) quota** across every
    leaderboard in the track without uploading.
  - `mode=release-upload` requires a protected GitHub Environment (`v2d-kaggle-release`)
    reviewer approval plus `confirm_weekly_quota_spend=CONFIRM_5_PER_WEEK_RULE`,
    then uploads the single verified file atomically across all leaderboards of
    the track and archives the `.kaggle-receipt.json` and `.v2d_submission_ledger.json`.

### 3. Enable the PreToolUse hook for coding agents (Claude Code / Codex / Cursor)

To prevent local coding agents or RSI loops from calling live Kaggle upload
endpoints in shell tools, register `agent_submission_guard.py` in `.claude/settings.json`
(or equivalent agent hook config):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|run_command|exec_command|edit_file|write_file",
        "hooks": [
          {
            "type": "command",
            "command": "python3 tools/v2d_compliance/agent_submission_guard.py"
          }
        ]
      }
    ]
  }
}
```

---

## CLI reference

### Audit a packed submission offline

```bash
python3 tools/v2d_compliance/compliance_gate.py audit \
  submissions/track3_submission.parquet \
  --track track_3 \
  --checkpoint-dir outputs/checkpoints \
  --repo-root . \
  --require-clean-git \
  --require-head-match \
  --check-quota \
  --receipt-out submissions/track3_submission.compliance.json
```

### Check rolling 7-day (5/week) quota for a track

```bash
python3 tools/v2d_compliance/compliance_gate.py check-quota --track track_3
```

### Scan scripts or test a shell command against the agent upload guard

```bash
python3 tools/v2d_compliance/agent_submission_guard.py --scan-scripts .
python3 tools/v2d_compliance/agent_submission_guard.py \
  --check-command "python3 submit/v2d_submit.py sub.parquet --track track_3 -m test --dry-run"
```
