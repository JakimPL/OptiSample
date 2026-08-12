import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    root = Path.cwd()
    while not (root / "pyproject.toml").exists() and root != root.parent:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import marimo as mo

    from notebooks.utils import labelling

    return Path, labelling, mo, root


@app.cell
def _(mo):
    mo.md("""
        # OptiSample — listening panel

        The answer sheet the fidelity metric is ranked against, filled one question at a time.

        For each question: **which of A and B sounds closer to the reference?** Answer on the scale, name
        what the side you rejected does wrong where a word comes to mind, and move on. Every answer is
        written to `labels.csv` as it is given, so the session survives a closed browser.

        Two answers place the sides level and say different things. *neither is closer* is for a pair where
        each side has its own fault and they cost the same; *no difference to hear* is for one you could
        find nothing to tell apart. The second holds the metric to a floor, so it is worth its own answer.

        Fixed volume and one pair of headphones throughout. Take the set in blocks with breaks between
        them, and answer each question as you hear it — a few of them are put twice, and how far those
        agree is what says whether the sheet can be trusted.
        """)
    return


@app.cell
def _(mo, root):
    listening_root = mo.ui.text(
        value=str(root / "artifacts" / "listening"), label="listening set root", full_width=True
    )
    instrument = mo.ui.text(value="Piano", label="instrument id")
    mo.vstack([mo.md("## Set"), listening_root, instrument])
    return instrument, listening_root


@app.cell
def _(mo):
    get_session, set_session = mo.state(None)
    get_place, set_place = mo.state(0)
    return get_place, get_session, set_place, set_session


@app.cell
def _(Path, instrument, labelling, listening_root, mo, set_place, set_session):
    def _open(_value):
        opened = labelling.open_session(Path(listening_root.value), instrument.value)
        set_session(opened)
        set_place(labelling.first_open(opened))

    open_button = mo.ui.button(label="open the set", on_click=_open)
    open_button
    return (open_button,)


@app.cell
def _(get_place, get_session, labelling, mo):
    session = get_session()
    place = get_place()
    mo.stop(session is None, mo.md("*Open a set to begin.*"))
    mo.md(
        f"**{place + 1} of {session.total}** — `{session.directories[place]}`  \n"
        f"{labelling.session_summary(session)}"
    )
    return place, session


@app.cell
def _(mo, place, session):
    _clips = session.clips(place)
    mo.vstack(
        [
            mo.md("### reference"),
            mo.audio(_clips.reference),
            mo.hstack(
                [
                    mo.vstack([mo.md("### A"), mo.audio(_clips.first)]),
                    mo.vstack([mo.md("### B"), mo.audio(_clips.second)]),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _(labelling, mo, place, session):
    _label = session.label(place)
    _verdicts = labelling.verdict_choices()
    _faults = labelling.fault_choices()
    verdict = mo.ui.dropdown(
        options=list(_verdicts),
        value=labelling.choice_label(_verdicts, _label.verdict),
        label="closer to the reference",
    )
    fault = mo.ui.dropdown(
        options=list(_faults),
        value=labelling.choice_label(_faults, _label.fault),
        label="what the rejected side does wrong",
    )
    note = mo.ui.text(value=_label.note, label="note", full_width=True)
    mo.vstack([verdict, fault, note])
    return fault, note, verdict


@app.cell
def _(fault, labelling, mo, note, place, session, set_place, set_session, verdict):
    def _answer(_value):
        answered = labelling.answer(
            session,
            place,
            verdict=labelling.verdict_choices()[verdict.value],
            fault=labelling.fault_choices()[fault.value],
            note=note.value,
        )
        set_session(answered)
        set_place(labelling.following(answered, place))

    def _step(offset):
        set_place((place + offset) % session.total)

    mo.hstack(
        [
            mo.ui.button(label="◀ back", on_click=lambda _value: _step(-1)),
            mo.ui.button(label="save and go on ▶", on_click=_answer),
            mo.ui.button(label="skip ▶", on_click=lambda _value: _step(1)),
        ]
    )
    return


@app.cell
def _(mo, session):
    mo.md(f"Answers are written to `{session.paths.labels_csv}` as they are given.")
    return


if __name__ == "__main__":
    app.run()
