"""Remote runner for Ampbox sandbox mode.

This module provides the RemoteRunner class for executing Amplifier sessions
on Ampbox infrastructure when the --sandbox flag is used.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import click

try:
    from ampbox_sdk import (
        AmpboxSessionClient,
        AuthenticationError,
        SessionNotFoundError,
        AmpboxError,
    )
except ImportError:
    # ampbox_sdk not installed - will provide helpful error message when used
    AmpboxSessionClient = None  # type: ignore
    AuthenticationError = Exception  # type: ignore
    SessionNotFoundError = Exception  # type: ignore
    AmpboxError = Exception  # type: ignore


class RemoteRunner:
    """Handles remote session execution via Ampbox."""

    def __init__(self):
        """Initialize the remote runner."""
        if AmpboxSessionClient is None:
            raise click.ClickException(
                "Ampbox SDK not installed. Install with:\n  pip install ampbox-sdk"
            )

        self.client = self._create_client()

    def _create_client(self):
        """Create Ampbox SDK client from config or environment."""
        api_url = os.getenv("AMPBOX_URL", "http://localhost:8080")
        api_key = os.getenv("AMPBOX_KEY")

        if not api_key:
            raise click.ClickException(
                "Ampbox API key required. Set via:\n"
                "  export AMPBOX_KEY=<your-key>\n"
                "  or configure in settings"
            )

        # Type checker knows AmpboxSessionClient is not None here due to __init__ check
        return AmpboxSessionClient(api_url=api_url, api_key=api_key)  # type: ignore

    def run(self, prompt: str, bundle: Optional[str] = None) -> None:
        """Create and run a remote session.

        Args:
            prompt: Initial prompt for the session
            bundle: Bundle to use (defaults to foundation:default)
        """
        bundle = bundle or "foundation:default"

        click.echo("🚀 Creating remote session on Ampbox...\n")

        session_id = None
        try:
            # Create session with current directory as codebase
            session = self.client.create_session(
                prompt=prompt,
                bundle=bundle,
                codebase_path=os.getcwd(),
            )
            session_id = session.id

            click.echo(f"✓ Session created: {session.id}")
            click.echo(f"✓ Bundle: {session.bundle}")
            click.echo(f"✓ Status: {session.status.value}\n")

            # Stream events from the session
            click.echo("Streaming events...\n")
            self._stream_events(session.id)

        except AuthenticationError:
            click.echo("✗ Authentication failed. Check your API key.", err=True)
            sys.exit(1)
        except AmpboxError as e:
            click.echo(f"✗ Ampbox error: {e}", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\n\n⏸️  Session paused.")
            if session_id:
                click.echo(f"Resume with: amplifier --sandbox resume {session_id}")
            sys.exit(0)

    def resume(self, session_id: str) -> None:
        """Resume a paused session.

        Args:
            session_id: ID of the session to resume
        """
        click.echo(f"🔄 Resuming session {session_id}...\n")

        try:
            session = self.client.resume_session(session_id)

            click.echo(f"✓ Session resumed: {session.id}")
            click.echo(f"✓ Status: {session.status.value}\n")

            # Stream events
            click.echo("Streaming events...\n")
            self._stream_events(session.id)

        except SessionNotFoundError:
            click.echo(f"✗ Session not found: {session_id}", err=True)
            sys.exit(1)
        except AmpboxError as e:
            click.echo(f"✗ Error: {e}", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            click.echo("\n\n⏸️  Session paused.")
            click.echo("Resume with: amplifier --sandbox resume {}".format(session_id))
            sys.exit(0)

    def list_sessions(self) -> None:
        """List user's sessions."""
        try:
            sessions = self.client.list_sessions()

            if not sessions:
                click.echo("No sessions found.")
                return

            # Display as table
            header = f"{'ID':<25} {'Status':<15} {'Bundle':<25} {'Created':<15}"
            click.echo(header)
            click.echo("-" * 85)
            for sess in sessions:
                click.echo(
                    f"{sess.id:<25} {sess.status.value:<15} {sess.bundle:<25} "
                    f"{sess.created_at.strftime('%Y-%m-%d %H:%M'):<15}"
                )

        except AmpboxError as e:
            click.echo(f"✗ Error: {e}", err=True)
            sys.exit(1)

    def _stream_events(self, session_id: str) -> None:
        """Stream events from session and display them.

        Args:
            session_id: ID of the session to stream from
        """
        for event in self.client.stream_events(session_id):
            # Handle different event types
            if event.type == "session:ready":
                click.echo("✓ Session initialized")

            elif event.type == "prompt:submit":
                click.echo("\n[New Turn]")
                click.echo(f"You: {event.data.get('prompt', '')}")

            elif event.type == "content:block:text":
                text = event.data.get("text", "")
                if text:
                    click.echo(text, nl=False)

            elif event.type == "tool:call":
                tool_name = event.data.get("tool_name", "unknown")
                click.echo(f"\n🔧 Using tool: {tool_name}")

            elif event.type == "tool:result":
                tool_name = event.data.get("tool_name", "unknown")
                success = event.data.get("success", False)
                if success:
                    click.echo(f"✓ {tool_name} completed")
                else:
                    error = event.data.get("error", "Unknown error")
                    click.echo(f"✗ {tool_name} failed: {error}")

            elif event.type == "session:completed":
                click.echo("\n\n✓ Session completed")
                break

            elif event.type == "session:failed":
                error = event.data.get("error", "Unknown error")
                click.echo(f"\n\n✗ Session failed: {error}")
                break


__all__ = ["RemoteRunner"]
