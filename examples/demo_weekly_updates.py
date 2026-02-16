#!/usr/bin/env python3
"""Demo script for weekly update scan functionality.

This demonstrates how to use the weekly_updates module to:
1. Scan for updated papers
2. Generate reports in markdown and HTML formats
"""

from datetime import datetime, timezone
from core.pipeline.weekly_updates import (
    scan_updated_papers,
    render_updates_md,
    render_updates_html,
    generate_update_report,
)


def main():
    """Run weekly update scan demo."""
    # Configuration
    db_path = "data/paper_scout.db"  # Adjust to your database path
    reference_date = datetime.now(timezone.utc).date().isoformat()
    output_dir = "tmp/reports"

    print("=" * 60)
    print("Weekly Update Scan Demo")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Reference Date: {reference_date}")
    print()

    # Method 1: Use the all-in-one function
    print("Method 1: Using generate_update_report()")
    print("-" * 60)
    result = generate_update_report(db_path, reference_date, output_dir)

    if result is None:
        print("No updated papers found for this week.")
    else:
        print(f"Reports generated in: {result}")
        print(f"  - {reference_date.replace('-', '')}_weekly_updates.md")
        print(f"  - {reference_date.replace('-', '')}_weekly_updates.html")
    print()

    # Method 2: Step-by-step approach
    print("Method 2: Step-by-step approach")
    print("-" * 60)

    # Step 1: Scan for papers
    papers = scan_updated_papers(db_path, reference_date)
    print(f"Found {len(papers)} updated paper(s)")
    print()

    if papers:
        # Show first paper details
        print("Sample paper:")
        paper = papers[0]
        print(f"  Title: {paper['title']}")
        print(f"  arXiv ID: {paper['native_id']}")
        print(f"  Score: {paper['llm_base_score']}")
        print(f"  Updated: {paper['updated_at_utc']}")
        print(f"  Topic: {paper['topic_slug']}")
        print()

        # Step 2: Generate markdown
        md_content = render_updates_md(papers, reference_date)
        print(f"Markdown report: {len(md_content)} characters")
        print()

        # Step 3: Generate HTML
        html_content = render_updates_html(papers, reference_date)
        print(f"HTML report: {len(html_content)} characters")
        print()

    print("=" * 60)
    print("Demo complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
