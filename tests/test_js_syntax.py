"""Verify that all inline JS in the HTML template is syntactically valid."""

import re
import shutil
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "negconv" / "gui" / "templates" / "index.html"


def _has_node():
    return shutil.which("node") is not None


def test_js_syntax_valid():
    """Extract all <script> blocks from the template and check with node."""
    if not _has_node():
        import pytest
        pytest.skip("node not available")

    html = TEMPLATE.read_text()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert len(blocks) >= 1, "No <script> blocks found in template"

    for i, js in enumerate(blocks):
        result = subprocess.run(
            ["node", "--check", "--input-type=module"],
            input=js,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"JS syntax error in <script> block {i + 1}:\n"
            f"{result.stderr.strip()}"
        )


def test_script_tags_balanced():
    """Every <script> has a matching </script>."""
    html = TEMPLATE.read_text()
    opens = html.count("<script>")
    closes = html.count("</script>")
    assert opens == closes, f"Unbalanced script tags: {opens} opens vs {opens} closes"


def test_no_template_syntax_in_js():
    """Jinja {{ }} or {% %} must not appear inside <script> blocks."""
    html = TEMPLATE.read_text()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    for i, js in enumerate(blocks):
        assert "{{" not in js, f"Jinja expression in <script> block {i + 1}"
        assert "{%" not in js, f"Jinja block in <script> block {i + 1}"
