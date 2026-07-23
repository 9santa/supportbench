from supportbench.retrieval.tokenization import tokenize


def test_tokenize_supports_latin() -> None:
    assert tokenize("OpenVPN GitHub") == [
        "openvpn",
        "github",
    ]


def test_tokenize_supports_cyrillic() -> None:
    assert tokenize("Настройка ВПН") == [
        "настройка",
        "впн",
    ]


def test_tokenize_lowercases_text() -> None:
    assert tokenize("VPN GitHub OpenVPN") == [
        "vpn",
        "github",
        "openvpn",
    ]


def test_tokenize_keeps_digits() -> None:
    assert tokenize("VPN2 2FA версия 1234") == [
        "vpn2",
        "2fa",
        "версия",
        "1234",
    ]


def test_tokenize_splits_punctuation() -> None:
    assert tokenize("VPN,GitHub.OpenVPN!2FA?") == [
        "vpn",
        "github",
        "openvpn",
        "2fa",
    ]


def test_tokenize_splits_hyphen() -> None:
    assert tokenize("VPN-клиент когда-нибудь") == [
        "vpn",
        "клиент",
        "когда",
        "нибудь",
    ]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_splits_underscores() -> None:
    assert tokenize("corporate_vpn gitlab_2fa") == [
        "corporate",
        "vpn",
        "gitlab",
        "2fa",
    ]
