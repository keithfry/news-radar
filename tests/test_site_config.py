"""Tests for the opt-in [site] keys that turn on a shared stylesheet and site bar."""

from __future__ import annotations

import shutil

from newsradar.config import load_config

from conftest import EXAMPLE_CONFIG


def test_example_config_leaves_site_chrome_off():
    site = load_config(EXAMPLE_CONFIG).site
    assert (site.stylesheet, site.home_url, site.home_label, site.section_label) == ("", "", "", "")


def test_site_chrome_keys_load(tmp_path):
    examples = tmp_path / "examples"
    shutil.copytree(EXAMPLE_CONFIG.parent, examples)
    cfg = examples / EXAMPLE_CONFIG.name
    text = cfg.read_text().replace(
        "[site]\n",
        "[site]\n"
        'stylesheet = "radar.css"\n'
        'home_url = "https://keithfry.net/"\n'
        'home_label = "Keith Fry"\n'
        'section_label = "Tech Radar"\n',
        1,
    )
    cfg.write_text(text)

    site = load_config(cfg).site
    assert site.stylesheet == "radar.css"
    assert site.home_url == "https://keithfry.net/"
    assert site.home_label == "Keith Fry"
    assert site.section_label == "Tech Radar"
