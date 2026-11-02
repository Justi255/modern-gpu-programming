"""Documentation configuration: deliberately independent of CUDA and PyTorch."""

project = "面向 RTX 5060 的现代 GPU 编程：从 Triton 到 FlashAttention"
author = "modern-gpu-programming contributors"
copyright = "2026, modern-gpu-programming contributors"
release = "0.1.0"
language = "zh_CN"
root_doc = "index"

extensions = ["myst_parser", "sphinx.ext.mathjax", "sphinx_copybutton"]
source_suffix = {".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3
myst_dmath_double_inline = False

# Keep missing references visible. A successful strict build should explain all
# sources, figures, and cross-references rather than suppressing their warnings.
nitpicky = True
exclude_patterns = ["_build"]

html_theme = "sphinx_book_theme"
html_title = "RTX 5060 · Triton GPU 编程"
html_static_path = ["static"]
html_theme_options = {
    "repository_url": "https://github.com/Justi255/modern-gpu-programming",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "show_toc_level": 2,
}
html_show_sourcelink = True
html_last_updated_fmt = "%Y-%m-%d"
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
pygments_style = "sphinx"
numfig = True
