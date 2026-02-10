import pytest

from letta.services.context_window_calculator.context_window_calculator import ContextWindowCalculator


def test_extract_system_components_git_backed_memory_without_memory_blocks_wrapper():
    system_message = """You are some system prompt.

<memory_filesystem>
Memory Directory: ~/.letta/agents/agent-123/memory

/memory/
└── system/
    └── human.md
</memory_filesystem>

<system/human.md>
---
description: test
limit: 10
---
hello
</system/human.md>

<memory_metadata>
- foo=bar
</memory_metadata>
"""

    system_prompt, core_memory, external_memory_summary = ContextWindowCalculator.extract_system_components(system_message)

    assert "You are some system prompt" in system_prompt
    assert "<memory_filesystem>" in core_memory
    assert "<system/human.md>" in core_memory
    assert external_memory_summary.startswith("<memory_metadata>")


def test_extract_system_components_legacy_memory_blocks_wrapper():
    system_message = """<base_instructions>SYS</base_instructions>

<memory_blocks>
<persona>p</persona>
</memory_blocks>

<memory_metadata>
- x=y
</memory_metadata>
"""

    system_prompt, core_memory, external_memory_summary = ContextWindowCalculator.extract_system_components(system_message)

    assert system_prompt.startswith("<base_instructions>")
    assert core_memory.startswith("<memory_blocks>")
    assert external_memory_summary.startswith("<memory_metadata>")
