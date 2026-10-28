"""The station visual: an animated, labelled bin board.

The demo must show *where the item goes*, not only a colour dot. This module draws
the six bins plus the review tray, lifts the lid of the chosen bin, and drops the
item into it. An item that goes to review is stopped before the bins instead.

It is drawn as one HTML block with CSS keyframes, so the browser does the animation
and Streamlit does no reruns. The block is keyed on the item id, so a new item
restarts the animation instead of showing the previous frame.
"""

from __future__ import annotations

from ecosort.actuation import ACTION_TEXT, BIN_COLOURS
from ecosort.routing import DESTINATION_LABELS, HUMAN_REVIEW

#: Bin order on the board, left to right.
BOARD_ORDER: list[str] = [
    "bin_cardboard",
    "bin_glass",
    "bin_metal",
    "bin_paper",
    "bin_plastic",
    "bin_trash",
]

#: Short labels printed on each bin lid.
BIN_SHORT: dict[str, str] = {
    "bin_cardboard": "Cardboard",
    "bin_glass": "Glass",
    "bin_metal": "Metal",
    "bin_paper": "Paper",
    "bin_plastic": "Plastic",
    "bin_trash": "General waste",
    HUMAN_REVIEW: "Human review",
}

#: One symbol per class, so the board is readable without colour alone.
BIN_GLYPH: dict[str, str] = {
    "bin_cardboard": "📦",
    "bin_glass": "🍾",
    "bin_metal": "🥫",
    "bin_paper": "📄",
    "bin_plastic": "🧴",
    "bin_trash": "🗑️",
    HUMAN_REVIEW: "🙋",
}


def _bin_html(destination: str, active: bool) -> str:
    colour = BIN_COLOURS.get(destination, "#616161")
    label = BIN_SHORT.get(destination, destination)
    glyph = BIN_GLYPH.get(destination, "•")
    classes = "bin active" if active else "bin"
    # The lid lifts only on the active bin; the body glows with the destination colour.
    return f"""
      <div class="{classes}" style="--bin:{colour}">
        <div class="lid"></div>
        <div class="body"><span class="glyph">{glyph}</span><span class="name">{label}</span></div>
      </div>"""


def render_bin_board(destination: str, item_hex: str, confidence: float,
                     is_review: bool) -> str:
    """Return the HTML for one actuation frame.

    Args:
        destination: the bin name, or ``HUMAN_REVIEW``.
        item_hex: a short id used to restart the animation for a new item.
        confidence: the classifier confidence, shown on the falling item.
        is_review: true when the item is stopped before the bins.
    """
    bins = "".join(_bin_html(name, name == destination) for name in BOARD_ORDER)
    review_active = "active" if is_review else ""
    review_colour = BIN_COLOURS[HUMAN_REVIEW]

    # Where the item lands: the review tray sits after the bins, so its offset is
    # the width of the board. The bins are laid out on a 7-column grid.
    index = BOARD_ORDER.index(destination) if destination in BOARD_ORDER else 6
    lane = index / 7 * 100

    action = ACTION_TEXT.get(destination, "Open flap")
    label = DESTINATION_LABELS.get(destination, destination)
    verdict = "stopped before the bins" if is_review else "into the bin"

    markup = f"""
<style>
  .board-wrap {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  .lane-stage {{ position: relative; height: 26px; margin: 0 0 2px 0; }}
  .falling {{
    position: absolute; left: {lane + 5.2:.3f}%; top: 0;
    width: 26px; height: 26px; border-radius: 50%;
    background: #ECEFF1; border: 2px solid #37474F;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: #263238;
    animation: drop-{item_hex} 1100ms cubic-bezier(.5,0,.9,.6) forwards;
  }}
  @keyframes drop-{item_hex} {{
    0%   {{ transform: translateY(-6px) scale(1.3); opacity: 0; }}
    20%  {{ opacity: 1; }}
    78%  {{ transform: translateY(16px) scale(0.9); opacity: 1; }}
    100% {{ transform: translateY(26px) scale(0.8); opacity: 0; }}
  }}
  .board {{ display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px; }}
  .bin {{ text-align: center; opacity: .45; transition: opacity 200ms ease; }}
  .bin .lid {{
    height: 12px; border-radius: 6px 6px 2px 2px; background: var(--bin);
    transform-origin: left bottom; transition: transform 260ms ease;
    display: flex; align-items: center; justify-content: center;
  }}
  .bin .body {{
    margin-top: 3px; padding: 10px 4px 12px; border-radius: 4px 4px 8px 8px;
    border: 2px solid var(--bin); background: color-mix(in srgb, var(--bin) 10%, white);
    min-height: 66px; display: flex; flex-direction: column; gap: 3px;
    align-items: center; justify-content: center;
  }}
  .bin .name {{ font-size: 10.5px; font-weight: 600; color: #37474F; line-height: 1.15; }}
  .bin .glyph {{ font-size: 20px; line-height: 1; filter: grayscale(1) opacity(.45); }}
  .bin.active .glyph {{ filter: none; }}
  .bin.active {{ opacity: 1; }}
  .bin.active .lid {{
    transform: rotate(-26deg) translateY(-4px);
    animation: lid-open-{item_hex} 320ms ease-out 900ms both;
  }}
  .bin.active .glyph {{ opacity: 1; }}
  .bin.active .body {{
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--bin) 35%, white), 0 6px 14px rgba(0,0,0,.13);
    animation: glow-{item_hex} 900ms ease-out 900ms both;
  }}
  @keyframes lid-open-{item_hex} {{ from {{ transform: none; }} }}
  @keyframes glow-{item_hex} {{
    from {{ box-shadow: 0 0 0 0 transparent; }}
    to   {{ box-shadow: 0 0 0 3px color-mix(in srgb, var(--bin) 35%, white),
                       0 6px 14px rgba(0,0,0,.13); }}
  }}
  .review {{ grid-column: 7; text-align: center; opacity: .45; }}
  .review .tray {{
    margin-top: 15px; padding: 10px 4px 12px; border-radius: 8px; min-height: 66px;
    border: 2px dashed {review_colour}; background: #FFF5F5;
    display: flex; align-items: center; justify-content: center;
  }}
  .review .name {{ font-size: 10.5px; font-weight: 600; color: {review_colour}; }}
  .review.active {{ opacity: 1; }}
  .review.active .tray {{
    box-shadow: 0 0 0 3px rgba(198,40,40,.22), 0 6px 14px rgba(0,0,0,.13);
    animation: pulse-{item_hex} 1.1s ease-in-out 900ms 2;
  }}
  @keyframes pulse-{item_hex} {{
    0%,100% {{ box-shadow: 0 0 0 3px rgba(198,40,40,.22); }}
    50%     {{ box-shadow: 0 0 0 7px rgba(198,40,40,.10); }}
  }}
  .verdict {{ margin-top: 8px; font-size: 12.5px; color: #263238; }}
  .verdict b {{ color: {BIN_COLOURS.get(destination, review_colour)}; }}
</style>

<div class="board-wrap">
  <div class="lane-stage">
    <div class="falling">{confidence * 100:.0f}</div>
  </div>
  <div class="board">
    {bins}
    <div class="review {review_active}">
      <div class="tray"><span class="name">Human<br>review</span></div>
    </div>
  </div>
  <div class="verdict">Item {item_hex} &rarr; <b>{label}</b> &middot; {verdict} &middot; {action}</div>
</div>
"""
    # Streamlit's markdown treats four or more leading spaces as a code block, which
    # prints the markup instead of drawing it. Flatten the indentation so the block
    # starts in column zero. CSS does not depend on indentation, so this is safe.
    return "\n".join(line.lstrip() for line in markup.strip().splitlines()) + "\n"
