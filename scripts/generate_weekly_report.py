#!/usr/bin/env python3
"""Generate weekly summary reports for Paper Scout.

Example usage:
    python3 scripts/generate_weekly_report.py --date 20240217
    python3 scripts/generate_weekly_report.py --date 20240217 --output reports/
"""

import argparse
from datetime import datetime
from pathlib import Path

from core.pipeline.weekly_summary import (
    generate_weekly_summary,
    render_weekly_summary_html,
    render_weekly_summary_md,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate weekly summary reports for Paper Scout"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date in YYYYMMDD format (end of week)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/paper_scout.db",
        help="Path to database file (default: data/paper_scout.db)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tmp/reports",
        help="Output directory (default: tmp/reports)",
    )

    args = parser.parse_args()

    # Validate date format
    try:
        date_obj = datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("   Expected: YYYYMMDD (e.g., 20240217)")
        return 1

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Generating weekly summary for {args.date}...")
    print(f"   Database: {args.db}")
    print(f"   Output: {output_dir}")

    # Generate summary data
    try:
        summary_data = generate_weekly_summary(args.db, args.date, str(output_dir))
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")
        return 1

    # Build weekly folder name using calendar year + ISO week
    ref_date = date_obj.date() if hasattr(date_obj, 'date') else date_obj
    _, iso_week, _ = ref_date.isocalendar()
    yy = f"{ref_date.year % 100:02d}"
    mm = f"{ref_date.month:02d}"
    ww = f"{iso_week:02d}"
    weekly_folder = output_dir / f"{yy}{mm}W{ww}_weekly_report"
    weekly_folder.mkdir(parents=True, exist_ok=True)

    # Render markdown
    md_output = weekly_folder / "report.md"
    md_content = render_weekly_summary_md(summary_data, args.date)
    md_output.write_text(md_content, encoding="utf-8")
    print(f"✅ Markdown: {md_output}")

    # Generate charts (if viz available)
    chart_paths = []
    try:
        from core.pipeline.weekly_viz import generate_weekly_charts, is_viz_available
        if is_viz_available():
            date_iso = date_obj.strftime("%Y-%m-%d")
            chart_paths = generate_weekly_charts(args.db, date_iso, str(weekly_folder))
            for cp in chart_paths:
                print(f"✅ Chart: {cp}")
    except Exception:
        pass

    # Render HTML
    html_output = weekly_folder / "report.html"
    html_content = render_weekly_summary_html(summary_data, args.date, chart_paths=chart_paths)
    html_output.write_text(html_content, encoding="utf-8")
    print(f"✅ HTML: {html_output}")

    # Print summary stats
    print("\n📈 Summary Statistics:")
    print(f"   Topics: {len(summary_data['keyword_freq'])}")
    print(f"   Top Papers: {len(summary_data['top_papers'])}")
    print(f"   Graduated Reminds: {len(summary_data['graduated_reminds'])}")
    print(f"   Score Trends: {sum(len(v) for v in summary_data['score_trends'].values())} data points")

    return 0


if __name__ == "__main__":
    exit(main())
