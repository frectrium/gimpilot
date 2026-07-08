from __future__ import annotations

import backend.conversation as conversation


def test_conversation_package_imports_cleanly():
    assert conversation.__doc__
