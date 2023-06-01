import os
import platform

import slack_sdk

from . import Logger


class Slack(object):
    """Slack messaging client.
    To connect to a workspace, you need to create an app (slack bot) and get its OAuth token.
    For more information, see https://api.slack.com/start/building/bolt-python

    Follow the steps below:
        1. Build your slack app. First, click "Create New App" at https://api.slack.com/apps and choose "From scratch".
            Type your app (slack bot) name and choose the workspace.
        2. Setup permission. Go to the "OAuth & Permissions" sidebar. Scroll down to the "Bot Token Scopes" and click "Add an OAuth Scope".
            Add the "chat:write" scope.
        3. Get OAuth token. Scroll up to the "OAuth Tokens for Your Workspace" and click "Install to Workspace". Then, copy the OAuth token.
        4. Enable the app in your slack channel. Choose your app on the setting page of your channel > "Integrations" > "Add apps".
        5. Add the token to the environment `$ export SLACK_BOT_TOKEN="xoxb-111-222-xxxxx"`
    """

    def __init__(self, channel):
        self.token = os.environ.get("SLACK_BOT_TOKEN")
        if self.token is None:
            Logger.warn(f"SLACK_BOT_TOKEN is not defined.\n\n{self.__doc__}")
            self.client = None
        else:
            self.client = slack_sdk.WebClient(token=self.token)
            self.channel = channel

    def msg(self, msg):
        if self.client is not None:
            self.client.chat_postMessage(
                channel=self.channel, text=f"{msg} @ {platform.node()}"
            )
        else:
            Logger.warn(
                f"Slack message is not delivered due to missing or wrong token (SLACK_BOT_TOKEN={self.token})"
            )

    def msg_and_wait(self, msg):
        self.msg(msg)
        return input(msg)
