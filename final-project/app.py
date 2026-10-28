"""EcoSort — User Interface sub-system.

A Streamlit front end for the integrated station. Four views, matching the way
the system is actually used:

1. **Sort an item** — capture a photo, see the classification, the confidence,
   the routing decision and the bin the actuator opened.
2. **Live dashboard** — contamination statistics read straight from the log.
3. **Batch evaluation** — run the held-out test set through the whole system,
   which is the end-to-end test from the test plan.
4. **About** — what the system is made of and why.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import streamlit as st  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ecosort.actuation import BIN_COLOURS
from ecosort.binboard import render_bin_board
from ecosort.capture import ImageCapture
from ecosort.classifier import (
    BackendUnavailableError,
    WasteClassifier,
    tensorflow_available,
)
from ecosort.config import MODEL_DIR, class_names, theta_for_backend
from ecosort.logging_db import WasteClass
from ecosort.pipeline import EcoSortSystem
from ecosort.routing import route
from ecosort.preprocess import InvalidImageError
from ecosort.routing import DESTINATION_LABELS

st.set_page_config(page_title="EcoSort — intelligent recycling sorting", page_icon="♻️", layout="wide")

CSS = """
<style>
  :root {
    --eco-green: #1B7F4B;
    --eco-leaf: #2E9E5B;
    --eco-mint: #E8F5EC;
    --eco-ink: #1F2A24;
    --eco-line: #D8E6DC;
  }
  .block-container { padding-top: 1.6rem; max-width: 1200px; }

  /* hero */
  .eco-hero {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 22px; margin-bottom: 6px;
    border-radius: 16px;
    background: linear-gradient(120deg, #E8F5EC 0%, #F6FBF7 60%, #FFFFFF 100%);
    border: 1px solid var(--eco-line);
  }
  .eco-hero .mark { font-size: 34px; line-height: 1; animation: sway 4s ease-in-out infinite; }
  @keyframes sway { 0%,100% { transform: rotate(-6deg); } 50% { transform: rotate(6deg); } }
  .eco-hero h1 { margin: 0; font-size: 1.7rem; color: var(--eco-ink); letter-spacing: -.02em; }
  .eco-hero p { margin: 2px 0 0; color: #4B5B52; font-size: .92rem; }

  /* live strip */
  .eco-strip { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0 4px; }
  .eco-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 999px; font-size: .82rem; font-weight: 600;
    background: var(--eco-mint); color: var(--eco-green); border: 1px solid var(--eco-line);
  }
  .eco-pill.warn { background: #FFF4E5; color: #9A5B00; border-color: #F2D9B4; }
  .eco-pill.bad  { background: #FDECEC; color: #A32020; border-color: #F3C7C7; }

  /* cards */
  .decision-card {
    border-radius: 16px; padding: 18px 20px; color: white; margin-bottom: 10px;
    box-shadow: 0 8px 22px rgba(27,127,75,.18);
    animation: rise 320ms ease-out both;
  }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } }
  .decision-card h2 { margin: 0 0 4px 0; font-size: 1.45rem; }
  .decision-card p { margin: 0; opacity: .95; font-size: .95rem; }
  .chip { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem;
          font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
          background: rgba(255,255,255,.22); }
  .muted { color: #6b7280; font-size: .88rem; }
  .bin-row { display: flex; align-items: center; gap: 10px; padding: 7px 0;
             border-bottom: 1px solid #eef1ec; font-size: .93rem; }
  .bin-dot { width: 13px; height: 13px; border-radius: 3px; flex: none; }
  .bin-active { font-weight: 700; }

  /* verdict banner */
  .eco-verdict { border-radius: 12px; padding: 12px 16px; margin: 8px 0; font-weight: 600;
                 animation: rise 260ms ease-out both; }
  .eco-verdict.good { background: #E8F5EC; color: #14663C; border-left: 4px solid #2E9E5B; }
  .eco-verdict.bad  { background: #FDECEC; color: #9B1C1C; border-left: 4px solid #C62828; }
  .eco-verdict.hold { background: #FFF4E5; color: #8A5200; border-left: 4px solid #E08A00; }

  /* tabs */
  .stTabs [data-baseweb="tab-list"] { gap: 4px; }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 8px 14px; font-weight: 600;
  }
  .stTabs [aria-selected="true"] { background: var(--eco-mint); color: var(--eco-green); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_classifier(backend: str, only_member: str | None = None):
    """Load the models once per (backend, artefact).

    The threshold is deliberately NOT part of this key. It used to be, which meant
    every movement of the slider reloaded two Keras models and made the page feel
    frozen. The model does not depend on the threshold, so it must not be reloaded.
    """
    return WasteClassifier(backend=backend, only_member=only_member)


def load_system(backend: str, theta: float, only_member: str | None = None) -> EcoSortSystem:
    """A thin wrapper around the cached classifier, with the current threshold."""
    return EcoSortSystem(
        backend=backend, theta=theta, only_member=only_member,
        classifier=load_classifier(backend, only_member),
    )


def system_available() -> tuple[bool, str]:
    """Check that at least one classifier artefact exists before building the system."""
    present = [name for name in ("mobilenetv3_224.keras", "mobilenetv2_224.keras",
                                 "svm_hog_glcm.pkl")
               if (MODEL_DIR / name).is_file()]
    if present:
        return True, ""
    return False, (
        f"No trained model found in `{MODEL_DIR}`. "
        "Run `kaggle/train_ecosort_224.ipynb`, then copy `mobilenetv3_224.keras` and "
        "`svm_hog_glcm.pkl` into `models/`."
    )


def decision_card(result) -> None:
    """The big routing answer, colour-coded to the destination."""
    decision = result.decision
    colour = BIN_COLOURS.get(decision.routed_to, "#616161")
    verdict = "HUMAN REVIEW" if decision.is_review else "ROUTED"
    st.markdown(
        f"""
        <div class="decision-card" style="background:{colour}">
          <span class="chip" style="background:rgba(255,255,255,.22)">{verdict}</span>
          <h2>{decision.destination_label}</h2>
          <p>{decision.reasoning}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def actuation_panel(result) -> None:
    """What the machine physically did: the item drops into the chosen bin."""
    st.markdown("##### Simulated actuation")
    # A short, stable token per item restarts the CSS animation for each new item.
    token = "".join(ch for ch in result.item_id if ch.isalnum())[-8:] or "item"
    st.markdown(
        render_bin_board(
            destination=result.decision.routed_to,
            item_hex=token,
            confidence=float(result.prediction.confidence),
            is_review=bool(result.decision.is_review),
        ),
        unsafe_allow_html=True,
    )


def probability_bars(prediction) -> None:
    """Full six-way distribution, so near-misses are visible."""
    st.markdown("##### Classifier confidence across all six classes")
    ordered = sorted(prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for label, probability in ordered:
        marker = " ●" if label == prediction.label else ""
        st.progress(min(max(probability, 0.0), 1.0), text=f"{label}{marker} — {probability:.1%}")


def next_item_id(system: EcoSortSystem) -> str:
    """A session-unique item id.

    The capture object restarts its counter at 1 in every new process, so two runs
    produced ``item_0001`` twice and the second preview overwrote the first. This
    counter starts above the highest id already in the log, so ids stay unique across
    restarts as well.
    """
    if "item_seq" not in st.session_state:
        highest = 0
        for entry in system.logger.entries:
            tail = entry.item_id.rsplit("_", 1)[-1]
            if tail.isdigit():
                highest = max(highest, int(tail))
        st.session_state.item_seq = highest
    st.session_state.item_seq += 1
    return f"item_{st.session_state.item_seq:04d}"


def render_sort_item(system: EcoSortSystem) -> None:
    st.subheader("Sort an item")
    st.caption(
        "Capture a photo of one waste item. The station classifies it, routes it, "
        "logs the decision and updates the dashboard."
    )

    left, right = st.columns([1, 1.25])

    with left:
        mode = st.radio(
            "Image capture",
            ["Upload a photo", "Use the camera", "Use a held-out sample"],
            horizontal=False,
            label_visibility="collapsed",
        )
        image_bytes = None
        true_class = None
        source_path = None

        if mode == "Upload a photo":
            upload = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
            if upload is not None:
                image_bytes = upload.getvalue()
        elif mode == "Use the camera":
            shot = st.camera_input("Take a photo of the item")
            if shot is not None:
                image_bytes = shot.getvalue()
        else:
            capture = ImageCapture()
            try:
                available = capture.available_samples("test")
            except (FileNotFoundError, KeyError) as exc:
                st.warning(f"Held-out samples unavailable: {exc}")
                available = 0
            if available:
                # All held-out items are selectable: the first twelve were an
                # arbitrary cap, and they hid most of the class distribution.
                if "sample_selector" not in st.session_state:
                    st.session_state.sample_selector = 0

                nav = st.columns([1, 1, 2])
                if nav[0].button("Previous", use_container_width=True):
                    st.session_state.sample_selector = (
                        st.session_state.sample_selector - 1) % available
                if nav[1].button("Next", use_container_width=True):
                    st.session_state.sample_selector = (
                        st.session_state.sample_selector + 1) % available
                if nav[2].button("Random item", use_container_width=True):
                    st.session_state.sample_selector = random.randrange(available)

                pick = st.selectbox(
                    "Held-out item",
                    options=list(range(available)),
                    format_func=lambda i: f"test item {i + 1} of {available}",
                    key="sample_selector",
                    help="Type to jump. Items are ordered by class, so you can browse one category at a time.",
                )
                item = capture.sample_items(count=1, split="test", start=pick)[0]
                image_bytes = _to_bytes(item.image)
                true_class = item.true_class
                # Keep the file path. Without it the logged item has no image_ref and
                # the review queue cannot show the photo again.
                source_path = item.source_path
                st.caption(f"Ground truth for this sample: **{true_class}**")

        if image_bytes is not None:
            st.image(image_bytes, caption="Captured item", use_container_width=True)

    with right:
        if image_bytes is None:
            st.info("Capture an item to see the classification and routing decision.")
            return

        # Unposed captures (an upload or a camera frame) get the robust multi-scale
        # pass. Held-out samples are studio images that already fill the frame, and
        # the extra passes would cost accuracy there.
        robust = mode != "Use a held-out sample"

        # A photo is classified ONCE. Streamlit reruns the whole script whenever any
        # widget changes, so without this guard every movement of the threshold
        # slider classified the same photo again and logged a duplicate row.
        signature = (
            hash(bytes(image_bytes)),
            system.classifier.backend,
            robust,
            true_class,
        )
        cached = st.session_state.get("last_item")
        if cached and cached.get("signature") == signature:
            result = cached["result"]
        else:
            try:
                result = system.process(
                    image_bytes, true_class=true_class, robust=robust,
                    item_id=next_item_id(system), source_path=source_path)
            except InvalidImageError as exc:
                st.error(f"Could not read that image: {exc}")
                return
            except BackendUnavailableError as exc:
                st.error(str(exc))
                return
            st.session_state.last_item = {"signature": signature, "result": result}

        # Keep the photo so the review queue can show it, keyed by the logged id.
        # Keep the photo for every review item so the queue can show it. A held-out
        # sample has a ground truth, but that is no reason to hide its picture.
        if result.decision.is_review:
            if "review_images" not in st.session_state:
                st.session_state.review_images = {}
            st.session_state.review_images[result.item_id] = image_bytes

        # The recorded decision was made at the threshold recorded with it. If the
        # slider has moved since, say so rather than letting the card look current.
        recorded_theta = result.decision.theta
        if abs(recorded_theta - system.theta) > 1e-9:
            live = route(result.prediction, theta=system.theta)
            st.warning(
                f"This item was recorded at θ = {recorded_theta:.0%}. The slider is now "
                f"{system.theta:.0%}, which would send it to **{live.destination_label}**"
                f"{' (a person)' if live.is_review else ''}. Process the item again to "
                f"record that decision."
            )

        decision_card(result)
        actuation_panel(result)

        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted class", result.prediction.label)
        metric_cols[1].metric("Confidence", f"{result.prediction.confidence:.1%}")
        metric_cols[2].metric("Latency", f"{result.latency_ms:.0f} ms")

        if result.was_correct is not None:
            if result.was_correct:
                st.markdown(
                    f'<div class="eco-verdict good">✅ Correct — the true class is '
                    f'<b>{result.captured.true_class}</b>.</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="eco-verdict bad">❌ Wrong bin avoided — the true class is '
                    f'<b>{result.captured.true_class}</b>. The item is logged, so the error '
                    f'shows on the dashboard.</div>',
                    unsafe_allow_html=True)
        elif result.decision.is_review:
            st.markdown(
                '<div class="eco-verdict hold">🙋 Held for a person — the station did not '
                'guess. Open the Review queue tab to decide.</div>',
                unsafe_allow_html=True)

        with st.expander("How the decision was made", expanded=False):
            st.write(
                f"**Backend:** {result.prediction.backend} · "
                f"**threshold θ:** {result.decision.theta:.0%} · "
                f"**robust mode:** {'on, 6 passes' if result.robust else 'off, 1 pass'}"
            )
            runner_up = result.prediction.runner_up
            if runner_up:
                st.write(
                    f"Closest alternative: **{runner_up[0]}** at {runner_up[1]:.1%}. "
                    f"{'That margin is why it went to review.' if result.decision.is_review else ''}"
                )
            st.json(result.summary())
            st.caption(
                "Per-stage timings (ms): "
                + ", ".join(f"{k} {v:.2f}" for k, v in result.timings_ms.items())
            )

        probability_bars(result.prediction)


def render_dashboard(system: EcoSortSystem) -> None:
    st.subheader("Live contamination dashboard")
    # ``system.summary()`` is the dashboard totals plus the backend and actuation
    # count; ``system.dashboard.summary()`` on its own does not carry those.
    summary = system.summary()
    total = summary["total_logged"]

    if total == 0:
        st.info("Nothing logged yet. Sort an item, or run the batch evaluation, to populate the dashboard.")
    else:
        cols = st.columns(4)
        cols[0].metric("Items logged", f"{total:,}")
        cols[1].metric(
            "Sent to human review",
            f"{summary['low_confidence_rate']:.1%}",
            help="Share of items the model was not confident enough to bin automatically.",
        )
        contamination = summary["contamination_rate"]
        cols[2].metric(
            "Contamination rate",
            "n/a — no ground truth" if contamination is None else f"{contamination:.1%}",
            help="Misclassification rate over evaluated items only. Live runs have no ground truth.",
        )
        auto = summary["auto_routed_accuracy"]
        cols[3].metric(
            "Auto-routed accuracy",
            "n/a" if auto is None else f"{auto:.1%}",
            help="Accuracy on the items the system was confident enough to bin itself.",
        )

        st.caption(
            f"Threshold θ = {system.theta:.0%} · active backend "
            f"`{summary['active_backend']}` · {summary['actuations']} actuations"
        )

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("##### Items per destination")
            st.bar_chart(summary["per_bin_counts"], height=260)
        with chart_cols[1]:
            st.markdown("##### Items per classified category")
            st.bar_chart(summary["per_class_counts"], height=260)

        trend = system.dashboard.timeseries()
        if not trend.empty:
            usable = trend.dropna(subset=["cumulative_accuracy"]).set_index("index")
            if not usable.empty:
                st.markdown("##### Running rates")
                trends = usable[["cumulative_accuracy", "review_rate"]].copy()
                trends.columns = ["accuracy (evaluated)", "review rate"]
                st.line_chart(trends, height=240)
            else:
                # A heading with nothing under it reads as a broken page. Say why the
                # chart is absent instead of leaving the space blank.
                st.info(
                    "No accuracy trend yet. The trend needs items with a known true "
                    "class, so run a batch evaluation or use a held-out sample."
                )

        per_class = summary["per_class_accuracy"]
        if any(row["evaluated"] for row in per_class.values()):
            st.markdown("##### Per-class accuracy (evaluated items only)")
            st.dataframe(
                [
                    {
                        "class": name,
                        "evaluated": row["evaluated"],
                        "correct": row["correct"],
                        "accuracy": f"{row['accuracy']:.1%}" if row["evaluated"] else "—",
                    }
                    for name, row in per_class.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("##### Recent decisions")
    recent = system.dashboard.recent(limit=15)
    if recent:
        st.dataframe(
            [
                {
                    "item": entry.item_id,
                    "predicted": entry.predicted_class.value,
                    "confidence": f"{entry.confidence:.1%}",
                    "destination": entry.routed_to,
                    "review": "yes" if entry.is_review else "no",
                    "backend": entry.backend,
                    "true class": entry.true_class.value if entry.true_class else "—",
                    "correct": "—" if entry.is_correct is None else ("yes" if entry.is_correct else "no"),
                }
                for entry in recent
            ],
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = _log_csv(system)
        st.download_button(
            "Download the decision log (CSV)",
            data=csv_bytes,
            file_name="ecosort_decision_log.csv",
            mime="text/csv",
        )
    else:
        st.caption("No decisions logged yet.")


#: What a reviewer should look at, per class. Short and practical: the review
#: queue is the one place a person replaces the model, so the advice must say what
#: separates the confusable classes rather than restate the class name.
REVIEW_GUIDE: dict[str, str] = {
    "glass": "Look for transparency and a reflection line. Glass is often green, brown or clear.",
    "metal": "Look for a solid, opaque surface with a sharp specular streak. Cans and foil are metal.",
    "plastic": "Look for a soft surface, a seam, or a recycling-number stamp.",
    "paper": "Look for a matte, uniform surface that folds easily. No corrugation.",
    "cardboard": "Look for a brown, corrugated edge. Tape and labels usually mean cardboard.",
    "trash": "Anything that is not one of the five recyclable materials, or is too dirty to recycle.",
}

#: The pair that the model confuses most, with the check that separates them.
CONFUSION_HINT: dict[frozenset, str] = {
    frozenset({"glass", "metal"}): (
        "Glass and metal confuse this model most. Check whether the item is see-through "
        "and whether it has a metal cap or lid."
    ),
    frozenset({"paper", "cardboard"}): (
        "Paper and cardboard confuse this model. Check the edge: a corrugated centre means cardboard."
    ),
}


def render_review(system: EcoSortSystem) -> None:
    """The human side of the routing decision.

    An item below theta goes to a person. Without this tab the queue would be a
    dead end: the dashboard could count reviews but nobody could clear them. Here
    an operator sees the photo, the model's guess, and six buttons.
    """
    st.subheader("Human review queue")
    st.caption(
        "Items the station was not confident enough to bin. A person decides the "
        "class, and the decision is written back to the log."
    )

    pending = system.logger.pending_reviews()
    resolved = [e for e in system.logger.entries
                if e.is_review and e.resolved_by is not None]

    cols = st.columns(3)
    cols[0].metric("Waiting for a person", len(pending))
    cols[1].metric("Resolved by a person", len(resolved))
    if resolved:
        agreed = sum(1 for e in resolved if e.predicted_class == e.true_class)
        cols[2].metric("Person agreed with the model", f"{agreed / len(resolved):.0%}")

    if not pending:
        st.info(
            "The queue is empty. Sort an item with a low confidence, or run a batch "
            "evaluation, to put something here."
        )
        return

    if "review_images" not in st.session_state:
        st.session_state.review_images = {}
    images = st.session_state.review_images
    entry = pending[0]

    left, right = st.columns([1, 1.3])
    with left:
        payload = images.get(entry.item_id)
        if payload is None and entry.image_ref:
            # The log points at the source photo, so the queue still works after a
            # restart when no in-memory preview exists.
            source = Path(entry.image_ref)
            if source.is_file():
                try:
                    payload = source.read_bytes()
                except OSError:
                    payload = None
        if payload:
            st.image(payload, caption=f"{entry.item_id} · {entry.backend}",
                     use_container_width=True)
            if entry.true_class is not None:
                st.caption(f"Ground truth for this sample: **{entry.true_class.value}**")
        else:
            # Say exactly why, so a missing photo is diagnosable rather than mysterious.
            if entry.image_ref:
                st.warning(f"The logged photo is missing on disk: `{entry.image_ref}`")
            else:
                st.warning(
                    "No photo is stored for this item. An uploaded photo has no file "
                    "path, and older log rows predate photo references. A held-out "
                    "sample or the batch evaluation always stores one. If this keeps "
                    "happening, press **Reset the log** and process items again."
                )
        # State the threshold that actually applied. "below theta" with no number
        # read as a contradiction when the slider had moved since the decision, and
        # {:.0%} rounding hid a value like 0.8995 behind "90%".
        if entry.theta is not None:
            st.caption(
                f"Model said **{entry.predicted_class.value}** at "
                f"**{entry.confidence:.1%}**. The threshold in force was "
                f"**{entry.theta:.0%}**, so the station held the item."
            )
        else:
            st.caption(
                f"Model said **{entry.predicted_class.value}** at "
                f"**{entry.confidence:.1%}**. This row predates threshold recording, "
                f"so the exact threshold is not stored."
            )

    with right:
        # A reviewer needs a recommendation, not just a number.
        runner = entry.runner_up
        st.markdown("##### Recommendation")
        second_line = (
            f" Its second choice was **{runner}**"
            + (f" at {entry.runner_up_confidence:.0%}" if entry.runner_up_confidence else "")
            + "." if runner else ""
        )
        # Use the threshold that applied to THIS item. The slider may have moved
        # since, and quoting the new value made the sentence contradict itself
        # ("78% confidence, which is below the 60% threshold").
        if entry.theta is None:
            st.markdown(
                f"The model identified this as **{entry.predicted_class.value}** with "
                f"**{entry.confidence:.1%}** confidence.{second_line} This row predates "
                f"threshold recording. Confirm the class below, and the station will "
                f"open that bin."
            )
        else:
            st.markdown(
                f"The model identified this as **{entry.predicted_class.value}** with "
                f"**{entry.confidence:.1%}** confidence, below the **{entry.theta:.0%}** "
                f"threshold that applied when it was sorted, so the station held it "
                f"instead of opening a bin.{second_line} Confirm the class below, and "
                f"the station will open that bin."
            )
        st.info(REVIEW_GUIDE.get(entry.predicted_class.value, ""))
        hint = CONFUSION_HINT.get(
            frozenset({entry.predicted_class.value, runner})) if runner else None
        if hint:
            st.warning(hint)

        st.markdown(f"##### Confirm the class for {entry.item_id}")
        classes = [c.value for c in WasteClass]
        for row in range(0, len(classes), 3):
            button_cols = st.columns(3)
            for col, class_name in zip(button_cols, classes[row:row + 3]):
                if col.button(class_name.title(), key=f"res_{entry.item_id}_{class_name}",
                              use_container_width=True):
                    decided = system.logger.resolve_review(
                        entry.item_id, WasteClass(class_name))
                    system.save_log()
                    # Show the item actually reaching a bin, which is the step the
                    # model could not take on its own.
                    st.success(
                        f"**{entry.item_id}** decided as **{class_name}**. "
                        f"The station now opens the "
                        f"**{DESTINATION_LABELS.get(decided.human_destination or '', 'bin')}**."
                    )
                    st.markdown(
                        render_bin_board(
                            destination=decided.human_destination or "human_review",
                            item_hex="".join(c for c in entry.item_id if c.isalnum())[-8:],
                            confidence=float(entry.confidence),
                            is_review=False,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.caption("This decision is saved to the log. Press any class to continue.")
                    st.stop()

        st.caption(
            f"{len(pending)} item(s) still waiting. Each decision is saved to "
            f"`{system.log_path.name}`."
        )

    if resolved:
        st.markdown("##### Decisions made by a person")
        st.dataframe(
            [{"item": e.item_id, "model said": e.predicted_class.value,
              "confidence": f"{e.confidence:.0%}", "person decided": e.true_class.value
              if e.true_class else "", "resolved by": e.resolved_by}
             for e in resolved],
            use_container_width=True, hide_index=True,
        )


def render_batch(system: EcoSortSystem) -> None:
    st.subheader("Batch evaluation — end-to-end test")
    st.caption(
        "Runs held-out items through the complete pipeline: capture → pre-process → "
        "classify → route → actuate → log → dashboard. This is the integration test "
        "from the test plan, driven from the UI."
    )

    controls = st.columns([1, 1, 2])
    count = controls[0].number_input("Items", min_value=1, max_value=380, value=40, step=10)
    run_all = controls[0].checkbox("Run all 380", value=False,
                                   help="Ignore the count and evaluate the whole held-out set.")
    split = controls[1].selectbox("Split", ["test", "val", "train"], index=0)
    backend = controls[2].selectbox(
        "Classifier backend",
        ["active", "cnn", "svm"],
        index=0,
        help="'active' uses whichever backend loaded successfully.",
    )

    if st.button("Run batch", type="primary"):
        capture = ImageCapture()
        wanted = 380 if run_all else int(count)
        try:
            items = capture.sample_items(count=wanted, split=split)
        except (FileNotFoundError, KeyError, InvalidImageError) as exc:
            st.error(f"Could not draw samples: {exc}")
            return

        progress = st.progress(0.0, text="Processing…")
        chosen = None if backend == "active" else backend
        results = []
        if "review_images" not in st.session_state:
            st.session_state.review_images = {}
        images = st.session_state.review_images
        for index, item in enumerate(items, start=1):
            try:
                produced = system.process(item.image, item_id=item.item_id,
                                          true_class=item.true_class, backend=chosen)
            except BackendUnavailableError as exc:
                st.error(str(exc))
                return
            results.append(produced)
            if produced.decision.is_review:
                buffer = io.BytesIO()
                item.image.save(buffer, format="PNG")
                images[produced.item_id] = buffer.getvalue()
            progress.progress(index / len(items), text=f"Processing… {index}/{len(items)}")
        progress.empty()

        correct = sum(1 for r in results if r.was_correct)
        review = sum(1 for r in results if r.decision.is_review)
        latencies = [r.latency_ms for r in results]

        cols = st.columns(4)
        cols[0].metric("Processed", len(results))
        cols[1].metric("Classification accuracy", f"{correct / len(results):.1%}")
        cols[2].metric("Sent to review", f"{review / len(results):.1%}")
        cols[3].metric("Mean latency", f"{sum(latencies) / len(latencies):.0f} ms")

        auto = [r for r in results if not r.decision.is_review]
        if auto:
            auto_correct = sum(1 for r in auto if r.was_correct)
            st.caption(
                f"Auto-routed accuracy: **{auto_correct / len(auto):.1%}** on {len(auto)} items "
                f"the system was confident enough to bin without a person."
            )
        st.caption(
            f"Slowest item {max(latencies):.0f} ms · requirement is ≤ 1000 ms per item."
        )
        st.dataframe([r.summary() for r in results], use_container_width=True, hide_index=True)


def render_about(system: EcoSortSystem) -> None:
    st.subheader("What this system is made of")
    st.markdown(
        """
EcoSort is an AI *system*, not a model. Seven sub-systems run in a fixed order,
and each one is a separate module in this repository:

| Sub-system | AI? | Module | Job |
|---|---|---|---|
| Image Capture | no | `ecosort/capture.py` | get a photo of the item |
| Pre-processing | no | `ecosort/preprocess.py` | resize to what each model expects |
| Waste Classifier | **yes** | `ecosort/classifier.py` | class + confidence |
| Decision / Bin Routing | no | `ecosort/routing.py` | bin, or a person, using θ |
| Logging | no | `ecosort/logging_db.py` | one record per decision |
| Dashboard | no | `ecosort/dashboard.py` | running contamination statistics |
| Simulated Actuation | no | `ecosort/actuation.py` | indicate the correct bin |
| User Interface | no | `app.py` | this screen |

The integration point is `ecosort/pipeline.py`: every item goes
capture → pre-process → classify → route → actuate → log, and the dashboard reads
only from the log, so the displayed totals cannot drift from what happened.
        """
    )

    st.markdown("##### How we got to this model")
    st.markdown(
        """
We tried six models. Each one answers a question that the model before it raised.
All six are scored on the same 380 held-out images, so the numbers are comparable.

| Step | Model | Accuracy | The question it answered |
| --- | --- | --- | --- |
| 1 | SVM on hand-crafted features | 0.711 | Can classic image features do the job? |
| 2 | Small CNN, random start | 0.692 | Do we need pretraining at all? |
| 3 | MobileNetV2, frozen backbone | 0.813 | Does pretraining help? |
| 4 | MobileNetV2, full fine-tune | 0.887 | Does the backbone need to adapt? |
| 5 | MobileNetV3, Recipe D | 0.913 | Is a newer backbone better? |
| 6 | **CNN blend (deployed)** | **0.921** | Can two models cover each other's faults? |
        """
    )

    st.markdown(
        """
**Step 1 to step 2.** We started with an SVM. It reads hand-made measurements: edge
direction, colour, and surface texture. It is fast and easy to explain. It reached
0.711. We then trained a small CNN from a random start. It reached 0.692, which is no
better. This step matters: it proves that the gain from later steps comes from
pretraining, not from the CNN shape.

**Step 2 to step 3.** MobileNetV2 comes with weights learned from 1.2 million images.
We first froze those weights and trained only a new final layer. This is the safe
way to use a pretrained model, because a large learning rate can destroy the learned
weights. The result was 0.813, a large jump.

**Step 3 to step 4.** A frozen backbone cannot change its features. We set the
backbone trainable at a small learning rate, so it could adapt to recycling photos
instead of the general images it had seen. The result was 0.887.

**Step 4 to step 5.** MobileNetV3 is a newer backbone in the same family. We copied a
public recipe for it: random resized crop, AdamW, label smoothing, a fixed budget of
12 epochs, and the final epoch's weights. The result was 0.913, the best single model.

**Step 5 to step 6.** MobileNetV2 and MobileNetV3 disagree on about one item in ten.
Their mistakes are not the same: one is stronger on some materials, the other on
others. So we average their two probability vectors. The interface you are using
calls this the **CNN blend**.

**Why the review queue exists.** Even the blend is right about 92 times in 100. A
wrong item in a recycling batch can reject the whole batch. So the station compares
the top probability with a threshold θ. If the value is below θ, the station does not
guess. It sends the item to a person.
        """
    )

    st.markdown("##### How the CNN blend works")
    st.markdown(
        """
For one photo, each model returns six probabilities. The blend multiplies each model
by a weight and adds the results:

```
final = 0.20 x MobileNetV2 + 0.80 x MobileNetV3
```

The weights were chosen on the validation set, not on the test set. MobileNetV3 gets
the larger weight because it is the more accurate single model. MobileNetV2 keeps a
share because it makes different mistakes.

The blend is better than either model alone: 0.887 and 0.913 become 0.921. It also
costs more time, because both models must run. That is why the sidebar lets you
select one model: you can compare the speed and the accuracy yourself.
        """
    )

    st.markdown("##### Why a confidence threshold")
    st.markdown(
        f"""
The classifier is right about 90 times in 100 on the held-out test set. That is
not good enough to bin every item unsupervised, because the errors are not evenly
spread: glass and metal look alike under reflection, and "trash" is a catch-all.

So the system does not guess. If the top class probability is below
**θ = {system.theta:.0%}**, the item is diverted to a person. θ is calibrated
separately for each classifier — MobileNetV2's probabilities are sharp, the SVM's
are blunt, so the SVM needs a lower threshold to route a comparable share of
items. The value comes from sweeping 0.30–0.90 on the validation set and taking
the most accurate automatic path that keeps the review queue inside a 10%
workload budget.
        """
    )

    info = system.classifier.info()
    st.markdown("##### Models in use")
    st.json(info)

    st.markdown("##### Environment")
    st.write(
        f"TensorFlow importable in this environment: **{'yes' if tensorflow_available() else 'no'}**"
    )
    if not tensorflow_available():
        st.caption(
            "The MobileNetV2 backend needs a working TensorFlow install. Where it is "
            "unavailable the system falls back to the SVM backend, which is the "
            "lightweight option the report recommends for constrained hardware."
        )


def _to_bytes(image) -> bytes:
    """Serialise a PIL image to PNG bytes for display and processing."""
    import io  # noqa: PLC0415

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _log_csv(system: EcoSortSystem) -> bytes:
    frame = system.logger.to_dataframe()
    return frame.to_csv(index=False).encode()


def main() -> None:
    st.markdown(
        """
        <div class="eco-hero">
          <div class="mark">♻️</div>
          <div>
            <h1>EcoSort</h1>
            <p>Point of disposal, sorted in one look. Capture, classify, route, log,
               and send only the uncertain items to a person.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ready, problem = system_available()
    if not ready:
        st.error(problem)
        st.stop()

    with st.sidebar:
        st.header("Station controls")
        MODEL_CHOICES = {
            "CNN blend — MobileNetV3 + MobileNetV2 (best)": ("cnn", None),
            "MobileNetV3-Large only (Recipe D)": ("cnn", "mobilenetv3_224.keras"),
            "MobileNetV2 only (Recipe B)": ("cnn", "mobilenetv2_224.keras"),
            "SVM on hand-crafted features (fallback)": ("svm", None),
        }
        chosen_label = st.selectbox(
            "Classifier",
            list(MODEL_CHOICES),
            index=0,
            help=("The blend is the best model. The other entries let you compare one "
                  "model at a time — the same four steps described in the About tab."),
        )
        backend, only_member = MODEL_CHOICES[chosen_label]
        theta = st.slider(
            "Confidence threshold θ",
            min_value=0.30, max_value=0.95, value=float(theta_for_backend(backend)), step=0.01,
            key=f"theta_{backend}",
            help=(
                "Below this confidence, an item goes to human review instead of a bin. "
                "Each backend is calibrated separately: the SVM's probabilities are "
                "blunter than MobileNetV2's, so it needs a lower threshold."
            ),
        )
        st.caption(
            f"Class names: {', '.join(class_names())}. "
            "The trained value is loaded from `models/metrics.json`."
        )

        try:
            system = load_system(backend, theta, only_member)
            active = system.classifier.backend
        except BackendUnavailableError as exc:
            st.error(str(exc))
            st.stop()
        if active != backend:
            st.warning(f"Fell back to the `{active}` backend: `{backend}` is unavailable here.")

        if st.button("Reset the log"):
            system.reset()
            for key in ("review_images", "last_item", "item_seq"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.caption(f"θ in use: {theta:.0%} · backend: `{active}`")

    summary = system.summary()
    waiting = len(system.logger.pending_reviews())
    logged = summary["total_logged"]
    backend_name = summary.get("active_backend", "?")
    accuracy = summary.get("accuracy_on_evaluated")
    strip = [
        f'<span class="eco-pill">🟢 backend {backend_name}</span>',
        f'<span class="eco-pill">θ {system.theta:.0%}</span>',
        f'<span class="eco-pill">📦 {logged} logged</span>',
    ]
    if accuracy is not None:
        strip.append(f'<span class="eco-pill">🎯 {accuracy:.0%} accurate</span>')
    if waiting:
        strip.append(f'<span class="eco-pill warn">🙋 {waiting} waiting for a person</span>')
    else:
        strip.append('<span class="eco-pill">✅ review queue empty</span>')
    st.markdown(f'<div class="eco-strip">{"".join(strip)}</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Sort an item", "Live dashboard", "Review queue",
                    "Batch evaluation", "About"])
    with tabs[0]:
        render_sort_item(system)
    with tabs[1]:
        render_dashboard(system)
    with tabs[2]:
        render_review(system)
    with tabs[3]:
        render_batch(system)
    with tabs[4]:
        render_about(system)


if __name__ == "__main__":
    main()
