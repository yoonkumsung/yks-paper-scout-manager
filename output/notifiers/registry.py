"""Notifier registry with topic-based routing.

Resolves per-topic NotifyConfig to concrete notifier instances by
looking up credentials from environment variables:

- Discord: ``DISCORD_WEBHOOK_{secret_key}``
- Telegram bot token: ``TELEGRAM_BOT_TOKEN_{secret_key}``
- Telegram chat ID: ``TELEGRAM_CHAT_ID_{secret_key}``

Section refs: DevSpec 11-1.
"""

from __future__ import annotations

import logging
import os
from typing import List

from core.models import NotifyConfig
from output.notifiers.base import NotifierBase

logger = logging.getLogger(__name__)


class NotifierRegistry:
    """Registry for notification providers with topic routing.

    Resolves per-topic provider + channel config to concrete notifier
    instances.  Maps ``secret_key`` to environment variable names:

    - Discord: ``DISCORD_WEBHOOK_{secret_key}``
    - Telegram bot token: ``TELEGRAM_BOT_TOKEN_{secret_key}``
    - Telegram chat ID: ``TELEGRAM_CHAT_ID_{secret_key}``
    """

    def get_notifier(self, notify_config: NotifyConfig) -> NotifierBase:
        """Resolve a NotifyConfig to a concrete notifier instance.

        Args:
            notify_config: Topic's notification configuration with
                provider, channel_id, secret_key.

        Returns:
            Configured notifier instance.

        Raises:
            ValueError: If provider is unknown or env var missing.
        """
        if notify_config.provider == "discord":
            env_key = f"DISCORD_WEBHOOK_{notify_config.secret_key}"
            webhook_url = os.environ.get(env_key, "")
            if not webhook_url:
                # Fallback: try DISCORD_WEBHOOK_URL (common convention)
                webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
            if not webhook_url:
                raise ValueError(f"Environment variable {env_key} not set")
            # Import here to avoid circular imports
            from output.notifiers.discord import DiscordNotifier

            return DiscordNotifier(webhook_url=webhook_url)

        elif notify_config.provider == "telegram":
            token_key = f"TELEGRAM_BOT_TOKEN_{notify_config.secret_key}"
            chat_key = f"TELEGRAM_CHAT_ID_{notify_config.secret_key}"
            bot_token = os.environ.get(token_key, "")
            chat_id = os.environ.get(chat_key, "")
            if not bot_token:
                raise ValueError(
                    f"Environment variable {token_key} not set"
                )
            if not chat_id:
                raise ValueError(
                    f"Environment variable {chat_key} not set"
                )
            from output.notifiers.telegram import TelegramNotifier

            return TelegramNotifier(bot_token=bot_token, chat_id=chat_id)

        else:
            raise ValueError(
                f"Unknown notification provider: {notify_config.provider}"
            )

    def get_notifiers_for_event(
        self, notify_configs: List[NotifyConfig], event: str
    ) -> List[NotifierBase]:
        """Return notifier instances for configs that include the given event.

        Args:
            notify_configs: List of notification configurations.
            event: Event type to filter on ("start" or "complete").

        Returns:
            List of configured notifier instances matching the event.
        """
        result: List[NotifierBase] = []
        for cfg in notify_configs:
            if event in cfg.events:
                try:
                    result.append(self.get_notifier(cfg))
                except ValueError as e:
                    logger.warning("Skipping notifier: %s", e)
        return result
