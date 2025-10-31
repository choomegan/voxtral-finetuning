"""
Helper functions for constructing chat-style prompts for tasks.
"""

from typing import List, Dict


def build_st_prompt(src_lang: str, audio_path: str) -> List[Dict]:
    """
    Build speech translation chat prompt with source language

    Translates src_lang -> English
    """
    source_lang = "Indonesian" if src_lang == "ind" else "Malay"

    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {
                    "type": "text",
                    "text": f"Translate this {source_lang} audio into English text.",
                },
            ],
        },
    ]
