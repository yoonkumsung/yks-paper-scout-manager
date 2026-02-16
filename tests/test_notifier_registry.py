"""Tests for output.notifiers.registry.NotifierRegistry.

Covers:
1. Discord provider resolves correctly with env var
2. Telegram provider resolves correctly with env vars
3. Missing Discord env var raises ValueError
4. Missing Telegram bot token raises ValueError
5. Missing Telegram chat ID raises ValueError
6. Unknown provider raises ValueError
7. Secret key mapping: DISCORD_WEBHOOK_{KEY} format correct
8. Secret key mapping: TELEGRAM_BOT_TOKEN_{KEY} / TELEGRAM_CHAT_ID_{KEY}
"""

from __future__ import annotations

import pytest

from core.models import NotifyConfig
from output.notifiers.registry import NotifierRegistry


# ---------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------


def _discord_config(secret_key: str = "MAIN") -> NotifyConfig:
    return NotifyConfig(
        provider="discord",
        channel_id="channel-123",
        secret_key=secret_key,
    )


def _telegram_config(secret_key: str = "MAIN") -> NotifyConfig:
    return NotifyConfig(
        provider="telegram",
        channel_id="chat-456",
        secret_key=secret_key,
    )


# ===============================================================
# 1. Discord provider resolves correctly with env var
# ===============================================================


class TestDiscordResolve:
    def test_discord_returns_discord_notifier(self, monkeypatch) -> None:
        monkeypatch.setenv("DISCORD_WEBHOOK_MAIN", "https://discord.com/api/webhooks/123/abc")
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_discord_config("MAIN"))
        from output.notifiers.discord import DiscordNotifier

        assert isinstance(notifier, DiscordNotifier)

    def test_discord_uses_correct_webhook_url(self, monkeypatch) -> None:
        url = "https://discord.com/api/webhooks/999/xyz"
        monkeypatch.setenv("DISCORD_WEBHOOK_TEAM_A", url)
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_discord_config("TEAM_A"))
        # Verify the internal URL is set (strip whitespace matching DiscordNotifier)
        assert notifier._webhook_url == url


# ===============================================================
# 2. Telegram provider resolves correctly with env vars
# ===============================================================


class TestTelegramResolve:
    def test_telegram_returns_telegram_notifier(self, monkeypatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_MAIN", "bot123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID_MAIN", "-100123456")
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_telegram_config("MAIN"))
        from output.notifiers.telegram import TelegramNotifier

        assert isinstance(notifier, TelegramNotifier)

    def test_telegram_uses_correct_credentials(self, monkeypatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_TEAM_B", "bot999:XYZ")
        monkeypatch.setenv("TELEGRAM_CHAT_ID_TEAM_B", "-200789")
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_telegram_config("TEAM_B"))
        assert notifier._bot_token == "bot999:XYZ"
        assert notifier._chat_id == "-200789"


# ===============================================================
# 3. Missing Discord env var raises ValueError
# ===============================================================


class TestMissingDiscordEnv:
    def test_missing_webhook_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("DISCORD_WEBHOOK_MAIN", raising=False)
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="DISCORD_WEBHOOK_MAIN"):
            registry.get_notifier(_discord_config("MAIN"))

    def test_empty_webhook_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("DISCORD_WEBHOOK_MAIN", "")
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="DISCORD_WEBHOOK_MAIN"):
            registry.get_notifier(_discord_config("MAIN"))


# ===============================================================
# 4. Missing Telegram bot token raises ValueError
# ===============================================================


class TestMissingTelegramToken:
    def test_missing_bot_token_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN_MAIN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID_MAIN", "-100123")
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN_MAIN"):
            registry.get_notifier(_telegram_config("MAIN"))


# ===============================================================
# 5. Missing Telegram chat ID raises ValueError
# ===============================================================


class TestMissingTelegramChatId:
    def test_missing_chat_id_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_MAIN", "bot:token")
        monkeypatch.delenv("TELEGRAM_CHAT_ID_MAIN", raising=False)
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="TELEGRAM_CHAT_ID_MAIN"):
            registry.get_notifier(_telegram_config("MAIN"))


# ===============================================================
# 6. Unknown provider raises ValueError
# ===============================================================


class TestUnknownProvider:
    def test_unknown_provider_raises(self) -> None:
        config = NotifyConfig(
            provider="slack",
            channel_id="C123",
            secret_key="MAIN",
        )
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="Unknown notification provider: slack"):
            registry.get_notifier(config)

    def test_empty_provider_raises(self) -> None:
        config = NotifyConfig(
            provider="",
            channel_id="C123",
            secret_key="MAIN",
        )
        registry = NotifierRegistry()
        with pytest.raises(ValueError, match="Unknown notification provider"):
            registry.get_notifier(config)


# ===============================================================
# 7. Secret key mapping: DISCORD_WEBHOOK_{KEY} format correct
# ===============================================================


class TestDiscordSecretKeyMapping:
    def test_key_with_underscores(self, monkeypatch) -> None:
        monkeypatch.setenv("DISCORD_WEBHOOK_MY_TEAM_PROD", "https://discord.com/api/webhooks/1/a")
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_discord_config("MY_TEAM_PROD"))
        from output.notifiers.discord import DiscordNotifier

        assert isinstance(notifier, DiscordNotifier)

    def test_key_format_includes_secret_key_verbatim(self, monkeypatch) -> None:
        """The env var name is exactly DISCORD_WEBHOOK_ + secret_key."""
        monkeypatch.setenv("DISCORD_WEBHOOK_XYZ123", "https://discord.com/api/webhooks/1/a")
        registry = NotifierRegistry()
        # Should not raise
        notifier = registry.get_notifier(_discord_config("XYZ123"))
        assert notifier is not None


# ===============================================================
# 8. Secret key mapping: TELEGRAM_BOT_TOKEN_{KEY} / TELEGRAM_CHAT_ID_{KEY}
# ===============================================================


class TestTelegramSecretKeyMapping:
    def test_key_with_complex_name(self, monkeypatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN_RESEARCH_LAB", "bot:abc")
        monkeypatch.setenv("TELEGRAM_CHAT_ID_RESEARCH_LAB", "-999")
        registry = NotifierRegistry()
        notifier = registry.get_notifier(_telegram_config("RESEARCH_LAB"))
        from output.notifiers.telegram import TelegramNotifier

        assert isinstance(notifier, TelegramNotifier)
        assert notifier._bot_token == "bot:abc"
        assert notifier._chat_id == "-999"
