"""Unit tests for src.mcp_server.html_converter."""

from __future__ import annotations

from src.mcp_server.html_converter import ImageRef, convert_html


class TestTableConversion:
    def test_simple_table(self):
        html = """
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>30</td></tr>
            <tr><td>Bob</td><td>25</td></tr>
        </table>
        """
        result = convert_html(html)
        assert "Name" in result.text
        assert "Alice" in result.text
        assert "|" in result.text
        # Separator line.
        assert "---" in result.text

    def test_empty_table(self):
        html = "<table></table>"
        result = convert_html(html)
        assert result.text.strip() == ""

    def test_uneven_columns(self):
        html = """
        <table>
            <tr><td>A</td><td>B</td><td>C</td></tr>
            <tr><td>1</td><td>2</td></tr>
        </table>
        """
        result = convert_html(html)
        # Should not crash; pads missing columns.
        assert "A" in result.text
        assert "1" in result.text


class TestCodeBlockConversion:
    def test_fenced_code_block(self):
        html = '<pre><code class="language-python">def hello():\n    pass</code></pre>'
        result = convert_html(html)
        assert "```python" in result.text
        assert "def hello():" in result.text
        assert result.text.strip().endswith("```")

    def test_no_language_hint(self):
        html = "<pre><code>some code</code></pre>"
        result = convert_html(html)
        assert "```\n" in result.text
        assert "some code" in result.text

    def test_highlight_class(self):
        html = '<pre><code class="highlight-javascript">let x = 1;</code></pre>'
        result = convert_html(html)
        assert "```javascript" in result.text

    def test_inline_code(self):
        html = "<p>Use <code>pip install</code> to install.</p>"
        result = convert_html(html)
        assert "`pip install`" in result.text


class TestMathConversion:
    def test_alttext_inline(self):
        html = '<math alttext="x^2 + y^2">ignored</math>'
        result = convert_html(html)
        assert "$x^2 + y^2$" in result.text

    def test_alttext_display(self):
        html = '<math display="block" alttext="E = mc^2">ignored</math>'
        result = convert_html(html)
        assert "$$E = mc^2$$" in result.text

    def test_structural_frac(self):
        html = """
        <math>
            <mfrac><mi>a</mi><mi>b</mi></mfrac>
        </math>
        """
        result = convert_html(html)
        assert r"\frac{a}{b}" in result.text

    def test_structural_sup(self):
        html = "<math><msup><mi>x</mi><mn>2</mn></msup></math>"
        result = convert_html(html)
        assert "x^{2}" in result.text

    def test_structural_sqrt(self):
        html = "<math><msqrt><mi>x</mi></msqrt></math>"
        result = convert_html(html)
        assert r"\sqrt{x}" in result.text

    def test_fallback_text(self):
        # No alttext, unknown structure → text extraction.
        html = "<math><mtext>some expression</mtext></math>"
        result = convert_html(html)
        assert "some expression" in result.text


class TestDefinitionList:
    def test_basic_dl(self):
        html = """
        <dl>
            <dt>Python</dt>
            <dd>A programming language.</dd>
            <dt>Rust</dt>
            <dd>A systems language.</dd>
        </dl>
        """
        result = convert_html(html)
        assert "**Python**" in result.text
        assert "A programming language." in result.text
        assert "**Rust**" in result.text


class TestImageExtraction:
    def test_basic_image(self):
        html = '<img src="https://example.com/img.png" alt="Diagram">'
        result = convert_html(html)
        assert len(result.images) == 1
        assert result.images[0].url == "https://example.com/img.png"
        assert result.images[0].alt == "Diagram"
        assert "[Diagram]" in result.text

    def test_relative_url_resolved(self):
        html = '<img src="/img/arch.png" alt="Architecture">'
        result = convert_html(html, base_url="https://example.com/docs/page.html")
        assert result.images[0].url == "https://example.com/img/arch.png"

    def test_figcaption(self):
        html = """
        <figure>
            <img src="https://example.com/chart.png" alt="Chart">
            <figcaption>A performance chart</figcaption>
        </figure>
        """
        result = convert_html(html)
        assert len(result.images) == 1
        assert result.images[0].caption == "A performance chart"

    def test_no_src_skipped(self):
        html = "<img alt='no source'>"
        result = convert_html(html)
        assert len(result.images) == 0


class TestAdmonitionConversion:
    def test_warning_admonition(self):
        html = '<div class="admonition warning"><p>Deprecated!</p></div>'
        result = convert_html(html)
        assert "WARNING" in result.text
        assert "Deprecated!" in result.text

    def test_note_admonition(self):
        html = '<div class="note"><p>Important info.</p></div>'
        result = convert_html(html)
        assert "NOTE" in result.text
        assert "Important info." in result.text

    def test_non_admonition_div(self):
        html = '<div class="container"><p>Hello</p></div>'
        result = convert_html(html)
        assert "Hello" in result.text
        # Should NOT be detected as admonition.
        assert "NOTE" not in result.text
        assert "WARNING" not in result.text


class TestFallbackStripping:
    def test_generic_html_stripped(self):
        html = "<span>Hello <b>world</b></span>"
        result = convert_html(html)
        assert "Hello" in result.text
        assert "world" in result.text
        assert "<span>" not in result.text
        assert "<b>" not in result.text

    def test_whitespace_collapsed(self):
        html = "<p>Line one</p>\n\n\n\n<p>Line two</p>"
        result = convert_html(html)
        # Should not have more than 2 consecutive newlines.
        assert "\n\n\n" not in result.text
        assert "Line one" in result.text
        assert "Line two" in result.text


class TestMixedContent:
    def test_table_and_code(self):
        html = """
        <table>
            <tr><th>Method</th><th>Time</th></tr>
            <tr><td>A</td><td>1ms</td></tr>
        </table>
        <pre><code class="language-python">print("hello")</code></pre>
        """
        result = convert_html(html)
        assert "|" in result.text  # Table pipes.
        assert "```python" in result.text
        assert 'print("hello")' in result.text

    def test_images_and_text(self):
        html = """
        <p>Here's a diagram:</p>
        <img src="https://example.com/d.png" alt="Flow">
        <p>And some explanation.</p>
        """
        result = convert_html(html)
        assert len(result.images) == 1
        assert "Flow" in result.text
        assert "explanation" in result.text
