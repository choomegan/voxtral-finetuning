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


def build_st_prompt_no_src_lang(audio_path: str) -> List[Dict]:
    """
    Build speech translation chat prompt without source language
    Translates unknown language -> English
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {
                    "type": "text",
                    "text": "Translate this audio into English text.",
                },
            ],
        },
    ]


def build_t2t_prompt_no_src_lang(src_text: str) -> List[Dict]:
    """
    Build text-to-text chat prompt without source language
    Translates unknown language -> English
    """
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Translate the following text into English. Only output the English translation. \n Input: {src_text}\n English:",
                },
            ],
        },
    ]
