from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "decision_room_app.py"


def test_shell_uses_content_height_iframe_for_one_page_scroll():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'height="content"' in source
    assert "height=1120" not in source


def test_cached_agent_perry_shell_renders():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)

    assert not app.exception
    assert app.title[0].value == "Agent Perry"
    assert app.text_input[0].value == "Haloperidol"
    assert app.text_input[1].value == "DRD2"
    assert [button.label for button in app.button] == [
        "Resolve molecule",
        "Launch live mission",
        "Restore judged demo",
    ]
    assert app.checkbox[0].label == "Use PubChem"
    assert any(
        "Bundled offline replay" in caption.value for caption in app.caption
    )


def test_resolve_catalog_molecule_without_network():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)

    app.text_input[0].set_value("Sulpiride")
    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert any("Resolved Sulpiride" in message.value for message in app.info)


def test_non_drd2_target_is_resolved_but_cannot_launch_without_a_model():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)

    app.text_input[0].set_value("Sulpiride")
    app.text_input[1].set_value("HTR2A")
    app.button[0].click().run(timeout=60)

    assert not app.exception
    assert any(
        "No HTR2A activity model is installed" in message.value
        for message in app.info
    )

    app.button[1].click().run(timeout=60)

    assert not app.exception
    assert any(
        "unsupported_target" in message.value
        for message in app.error
    )


def test_restore_demo_returns_to_haloperidol():
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=60)
    app.text_input[0].set_value("Sulpiride")
    app.text_input[1].set_value("HTR2A")
    app.button[0].click().run(timeout=60)

    app.button[2].click().run(timeout=60)

    assert not app.exception
    assert app.text_input[0].value == "Haloperidol"
    assert app.text_input[1].value == "DRD2"
    assert any(
        "Bundled offline replay" in caption.value for caption in app.caption
    )
