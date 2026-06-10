"""Single-click browser download — the ONE implementation.

Streamlit's ``st.download_button`` requires its own click; when the user
already clicked a "Generate" button, this injects a hidden ``<a download>``
anchor + auto-click script so generating = downloading (house rule: never a
two-step Generate-then-Download).

Previously copy-pasted in four places (header_impact, step2_review_ipr,
sidebar, well_sort) — those now delegate here.
"""

import base64

import streamlit.components.v1 as components


def autodownload(
    data: bytes, filename: str, mime: str, dom_id: str | None = None
) -> None:
    """Trigger an immediate browser download of ``data``.

    Args:
        data: file bytes.
        filename: download filename (also seeds the anchor's DOM id).
        mime: MIME type (e.g. "application/pdf", "image/png").
        dom_id: optional explicit DOM id when several downloads can render
            on one page and must not collide.
    """
    dom = dom_id or "dl_" + "".join(c if c.isalnum() else "_" for c in filename)
    b64 = base64.b64encode(data).decode()
    components.html(
        f'<a id="{dom}" download="{filename}" '
        f'href="data:{mime};base64,{b64}" style="display:none"></a>'
        f'<script>document.getElementById("{dom}").click()</script>',
        height=0,
    )
