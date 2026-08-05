"""The deployment's basic-auth guard.

The guard exists because the hosted app exposes a 200 MB file upload that
spawns a torch subprocess, so these tests care most about the two ways a guard
fails open: being off when it was meant to be on, and accepting a credential it
should not.
"""

from __future__ import annotations

import base64

import pytest
from flask import Flask

from bridge_rna.auth import (
    HEALTH_PATH,
    check,
    credential_from_env,
    install_basic_auth,
    parse_credential,
)


def _header(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return f"Basic {token}"


@pytest.mark.parametrize(
    "raw",
    ["", "nocolon", ":password", "user:", ":", "   "],
)
def test_unusable_credentials_are_treated_as_unset(raw):
    """A blank half must disable the guard, never become a guessable password."""
    assert parse_credential(raw.strip()) is None


def test_a_well_formed_credential_parses():
    assert parse_credential("ames:orbit") == ("ames", "orbit")


def test_a_password_may_contain_colons():
    """Only the first colon separates, so a colon in the password survives."""
    assert parse_credential("ames:a:b:c") == ("ames", "a:b:c")


def test_credential_comes_from_the_environment():
    assert credential_from_env({"BRIDGE_RNA_BASIC_AUTH": "ames:orbit"}) == (
        "ames",
        "orbit",
    )
    assert credential_from_env({}) is None


class TestCheck:
    expected = ("ames", "orbit")

    def test_the_right_credential_passes(self):
        assert check(_header("ames", "orbit"), self.expected) is True

    @pytest.mark.parametrize(
        "header",
        [
            None,
            "",
            _header("ames", "wrong"),
            _header("wrong", "orbit"),
            _header("", ""),
            "Basic",
            "Basic !!!not-base64!!!",
            "Bearer " + base64.b64encode(b"ames:orbit").decode(),
            base64.b64encode(b"ames:orbit").decode(),  # no scheme
            "Basic " + base64.b64encode(b"nocolon").decode(),
        ],
    )
    def test_everything_else_fails(self, header):
        assert check(header, self.expected) is False

    def test_a_prefix_of_the_password_is_rejected(self):
        """Guards against a comparison that stops at the shorter string."""
        assert check(_header("ames", "orb"), self.expected) is False


class TestInstall:
    @staticmethod
    def _app():
        server = Flask(__name__)

        @server.route("/")
        def index():
            return "the app"

        @server.route(HEALTH_PATH)
        def healthz():
            return "ok"

        return server

    def test_no_credential_leaves_the_app_open_and_reports_it(self):
        server = self._app()
        assert install_basic_auth(server, credential=None) is False
        assert server.test_client().get("/").status_code == 200

    def test_a_guarded_app_challenges_an_anonymous_request(self):
        server = self._app()
        assert install_basic_auth(server, credential=("ames", "orbit")) is True
        response = server.test_client().get("/")
        assert response.status_code == 401
        # Without this header a browser never offers a login prompt.
        assert response.headers["WWW-Authenticate"].startswith("Basic realm=")

    def test_a_guarded_app_admits_the_right_credential(self):
        server = self._app()
        install_basic_auth(server, credential=("ames", "orbit"))
        response = server.test_client().get(
            "/", headers={"Authorization": _header("ames", "orbit")}
        )
        assert response.status_code == 200
        assert b"the app" in response.data

    def test_a_guarded_app_rejects_a_wrong_password(self):
        server = self._app()
        install_basic_auth(server, credential=("ames", "orbit"))
        response = server.test_client().get(
            "/", headers={"Authorization": _header("ames", "wrong")}
        )
        assert response.status_code == 401

    def test_the_health_path_stays_reachable_without_a_credential(self):
        """Fly's health check has no credential to present."""
        server = self._app()
        install_basic_auth(server, credential=("ames", "orbit"))
        response = server.test_client().get(HEALTH_PATH)
        assert response.status_code == 200

    def test_the_guard_covers_more_than_the_index(self):
        """A path-by-path allowlist would have missed the upload endpoint."""
        server = self._app()

        @server.route("/_dash-update-component", methods=["POST"])
        def update():
            return "{}"

        install_basic_auth(server, credential=("ames", "orbit"))
        assert server.test_client().post("/_dash-update-component").status_code == 401
