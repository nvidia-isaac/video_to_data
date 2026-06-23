# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
HTML report generator for pipeline results.

Creates an interactive HTML page to browse clips and verification results.
Operates on ClipContext objects with absolute timestamps.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from video_ingestion_agent.ingestion.state import ClipContext, VerificationResult

logger = logging.getLogger(__name__)


def generate_html_report(
    clips: list[ClipContext],
    run_dir: Path,
    verifications: list[VerificationResult] | None = None,
    config_summary: dict[str, Any] | None = None,
) -> Path:
    """
    Generate interactive HTML report for pipeline results.

    Args:
        clips: List of ClipContext objects
        run_dir: Run directory
        verifications: Optional verification results
        config_summary: Optional configuration summary

    Returns:
        Path to generated HTML file
    """
    logger.info("Generating HTML report...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data for JavaScript
    clips_data = [clip.model_dump() for clip in clips]

    # Build verification map
    verification_map = {v.clip_id: v.model_dump() for v in verifications} if verifications else {}

    # Statistics
    total_clips = len(clips)
    video_files = sorted({clip.video_path for clip in clips})

    if verifications:
        valid_clips = sum(1 for v in verifications if v.is_valid)
        invalid_clips = sum(1 for v in verifications if not v.is_valid)
    else:
        valid_clips = invalid_clips = 0

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Pipeline Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto; background: white;
            border-radius: 16px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .subtitle {{ font-size: 1.1em; opacity: 0.9; }}
        .stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; padding: 30px 40px; background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        .stat-card {{
            background: white; padding: 20px; border-radius: 12px;
            text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-value {{ font-size: 2.5em; font-weight: 700; color: #667eea; }}
        .stat-label {{ color: #666; font-size: 0.9em; text-transform: uppercase; }}
        .controls {{ padding: 30px 40px; background: white; border-bottom: 1px solid #e0e0e0; }}
        .filter-group {{ margin-bottom: 20px; }}
        .filter-group label {{ display: block; font-weight: 600; color: #333; margin-bottom: 8px; }}
        .filter-group select, .filter-group input {{
            width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 1em;
        }}
        .clips-grid {{
            padding: 40px; display: grid;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 30px;
        }}
        .clip-card {{
            background: white; border-radius: 12px; overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 2px solid transparent;
            transition: all 0.3s;
        }}
        .clip-card:hover {{ transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.15); }}
        .clip-card.valid {{ border-color: #4caf50; }}
        .clip-card.invalid {{ border-color: #f44336; }}
        .clip-info {{ padding: 20px; }}
        .clip-header {{ display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px; }}
        .clip-id {{ font-weight: 700; color: #333; font-size: 1.1em; }}
        .clip-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .badge-valid {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-invalid {{ background: #ffebee; color: #c62828; }}
        .meta-row {{ display: flex; margin-bottom: 8px; font-size: 0.9em; }}
        .meta-label {{ font-weight: 600; color: #666; min-width: 100px; }}
        .meta-value {{ color: #333; flex: 1; }}
        .clip-description {{
            color: #555; line-height: 1.6; margin-top: 15px;
            padding-top: 15px; border-top: 1px solid #e0e0e0; font-size: 0.95em;
        }}
        .verification-details {{
            margin-top: 15px; padding: 15px; background: #f8f9fa;
            border-radius: 8px; font-size: 0.9em;
        }}
        .issues-list {{ margin-top: 10px; padding-left: 20px; }}
        .issues-list li {{ margin-bottom: 4px; color: #c62828; }}
        .footer {{ background: #f8f9fa; padding: 20px 40px; text-align: center; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Video Pipeline Report</h1>
            <p class="subtitle">Generated on {timestamp}</p>
        </div>
        <div class="stats">
            <div class="stat-card"><div class="stat-value">{total_clips}</div><div class="stat-label">Total Clips</div></div>
            <div class="stat-card"><div class="stat-value">{len(video_files)}</div><div class="stat-label">Source Videos</div></div>
            {f'<div class="stat-card"><div class="stat-value">{valid_clips}</div><div class="stat-label">Valid Clips</div></div><div class="stat-card"><div class="stat-value">{invalid_clips}</div><div class="stat-label">Invalid Clips</div></div>' if verifications else ""}
        </div>
        <div class="controls">
            <div class="filter-group">
                <label for="videoFilter">Filter by Video:</label>
                <select id="videoFilter"><option value="">All Videos</option>
                {chr(10).join(f'<option value="{vf}">{vf}</option>' for vf in video_files)}
                </select>
            </div>
            {'''<div class="filter-group"><label for="statusFilter">Filter by Status:</label><select id="statusFilter"><option value="">All</option><option value="valid">Valid</option><option value="invalid">Invalid</option></select></div>''' if verifications else ""}
            <div class="filter-group">
                <label for="searchBox">Search (object, action, description):</label>
                <input type="text" id="searchBox" placeholder="Type to search...">
            </div>
        </div>
        <div class="clips-grid" id="clipsGrid"></div>
        <div class="footer">
            <p>Video Ingestion Agent | Run Dir: {run_dir}</p>
            <p>{json.dumps(config_summary or {}, default=str)}</p>
        </div>
    </div>
    <script>
        const clipsData = {json.dumps(clips_data, indent=2, default=str)};
        const verificationsMap = {json.dumps(verification_map, indent=2, default=str)};
        const hasVerifications = {json.dumps(bool(verifications))};

        function renderClips(clips) {{
            const grid = document.getElementById('clipsGrid');
            if (clips.length === 0) {{
                grid.innerHTML = '<div style="grid-column:1/-1;text-align:center;padding:60px;color:#666;">No clips match your filters</div>';
                return;
            }}
            grid.innerHTML = clips.map(clip => {{
                const v = verificationsMap[clip.clip_id];
                const isValid = v ? v.is_valid : null;
                const validClass = isValid === true ? 'valid' : (isValid === false ? 'invalid' : '');
                return `
                    <div class="clip-card ${{validClass}}">
                        <div class="clip-info">
                            <div class="clip-header">
                                <div class="clip-id">${{clip.clip_id}}</div>
                                ${{isValid !== null ? `<span class="clip-badge badge-${{isValid ? 'valid' : 'invalid'}}">${{isValid ? 'Valid' : 'Invalid'}}</span>` : ''}}
                            </div>
                            <div>
                                <div class="meta-row"><span class="meta-label">Video:</span><span class="meta-value">${{clip.video_path}}</span></div>
                                <div class="meta-row"><span class="meta-label">Time:</span><span class="meta-value">${{clip.start_t.toFixed(1)}}s - ${{clip.end_t.toFixed(1)}}s (${{(clip.end_t - clip.start_t).toFixed(1)}}s)</span></div>
                                <div class="meta-row"><span class="meta-label">Object:</span><span class="meta-value">${{clip.object || 'N/A'}}</span></div>
                                <div class="meta-row"><span class="meta-label">Action:</span><span class="meta-value">${{clip.action || 'N/A'}}</span></div>
                            </div>
                            ${{clip.description ? `<div class="clip-description">${{clip.description}}</div>` : ''}}
                            ${{v && !v.is_valid ? `
                                <div class="verification-details">
                                    <strong>Score:</strong> ${{v.verification_score.toFixed(2)}}
                                    ${{v.violations && v.violations.length > 0 ? `<ul class="issues-list">${{v.violations.map(i => `<li>${{i}}</li>`).join('')}}</ul>` : ''}}
                                </div>
                            ` : ''}}
                        </div>
                    </div>`;
            }}).join('');
        }}

        function filterClips() {{
            const videoFilter = document.getElementById('videoFilter').value;
            const statusEl = document.getElementById('statusFilter');
            const statusFilter = statusEl ? statusEl.value : '';
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            let filtered = clipsData;
            if (videoFilter) filtered = filtered.filter(c => c.video_path === videoFilter);
            if (statusFilter && hasVerifications) {{
                filtered = filtered.filter(c => {{
                    const v = verificationsMap[c.clip_id];
                    if (!v) return false;
                    return statusFilter === 'valid' ? v.is_valid : !v.is_valid;
                }});
            }}
            if (searchTerm) {{
                filtered = filtered.filter(c => {{
                    return [c.object||'', c.action||'', c.description||'', c.clip_id].join(' ').toLowerCase().includes(searchTerm);
                }});
            }}
            renderClips(filtered);
        }}

        document.getElementById('videoFilter').addEventListener('change', filterClips);
        if (hasVerifications) document.getElementById('statusFilter').addEventListener('change', filterClips);
        document.getElementById('searchBox').addEventListener('input', filterClips);
        renderClips(clipsData);
    </script>
</body>
</html>
"""

    output_path = run_dir / "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved to: {output_path}")
    return output_path
