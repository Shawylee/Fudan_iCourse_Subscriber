"""Offline tests for the UIS second-factor flow."""

import unittest
from unittest.mock import patch

import main
from src.api.webvpn import SecondFactorError, WebVPNSession, get_vpn_url, totp_code
from src.runtime import config


# Base32 encoding of the public RFC 6238 SHA-1 test key.
TEST_KEY = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"


class UISSecondFactorTests(unittest.TestCase):
    def test_totp_matches_rfc_vector_truncated_to_six_digits(self):
        self.assertEqual(totp_code(TEST_KEY, 59), "287082")

    def test_rejects_missing_or_invalid_secret(self):
        for secret in ("", "123456"):
            with self.subTest(secret=secret), self.assertRaises(SecondFactorError):
                totp_code(secret, 59)

    def test_submits_totp_to_the_correct_uis_endpoint(self):
        challenge = {
            "moduleCodes": ["userAndOtp"],
            "authChainCode": "chain",
            "requestNumber": "request",
        }
        token = {"code": 200, "loginToken": "test-token"}
        for via_vpn in (False, True):
            vpn = WebVPNSession()
            with self.subTest(via_vpn=via_vpn), vpn.session as session:
                with patch.object(config, "TOTP_SECRET", TEST_KEY), patch.object(
                    session, "post"
                ) as post, patch("src.api.webvpn.time.time", return_value=59):
                    post.return_value.json.return_value = token
                    self.assertEqual(
                        vpn._authenticate_totp(
                            challenge, "student", "lck", "entity", "type", via_vpn=via_vpn
                        ),
                        token,
                    )
                    base = get_vpn_url(config.IDP_BASE) if via_vpn else config.IDP_BASE
                    self.assertEqual(post.call_args.args[0], f"{base}/idp/authn/authExecute")
                    self.assertEqual(post.call_args.kwargs["json"]["authPara"]["otpCode"], "287082")

    def test_does_not_retry_when_second_factor_needs_attention(self):
        with patch("main.WebVPNSession") as factory, patch("main.time.sleep") as sleep:
            factory.return_value.authenticate_icourse.side_effect = SecondFactorError("test")
            with self.assertRaises(SecondFactorError):
                main.login_with_retry()
            factory.assert_called_once()
            factory.return_value.session.close.assert_called_once()
            sleep.assert_not_called()
