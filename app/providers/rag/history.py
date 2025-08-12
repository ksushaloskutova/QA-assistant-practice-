from __future__ import annotations

from typing import Dict, List, Union

from langchain_core.messages import AIMessage, HumanMessage

_chat_history: Dict[str, List[Union[HumanMessage, AIMessage]]] = {}


def get_session_history(session_id: str) -> List[Union[HumanMessage, AIMessage]]:
    if session_id not in _chat_history:
        _chat_history[session_id] = []
    return _chat_history[session_id]


def append_pair(session_id: str, human_text: str, ai_text: str) -> None:
    hist = get_session_history(session_id)
    hist.append(HumanMessage(content=human_text))
    hist.append(AIMessage(content=ai_text))
