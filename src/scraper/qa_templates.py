"""Vietnamese question templates and ChatML pair builder for training data generation."""

SYSTEM_PROMPT_VI = (
    "Bạn là tư vấn viên thông tin đào tạo của Trường Đại học Công nghệ Thông tin "
    "- ĐHQG TP.HCM (UIT). Hãy trả lời câu hỏi của sinh viên một cách chính xác và thân thiện."
)

# Question templates grouped by intent
QUESTION_TEMPLATES: list[str] = [
    # Informational
    "{topic} là gì?",
    "Cho tôi biết thông tin về {topic}",
    "Giới thiệu về {topic}",
    "{topic} bao gồm những gì?",

    # Procedural
    "Làm thế nào để {action}?",
    "Hướng dẫn {action} như thế nào?",
    "Các bước để {action}?",
    "Quy trình {process} như thế nào?",
    "Thủ tục {procedure} gồm những gì?",

    # Requirements
    "Điều kiện {requirement} là gì?",
    "Cần những gì để {requirement}?",
    "Yêu cầu {requirement} bao gồm những gì?",

    # Time/Schedule
    "Khi nào thì {event}?",
    "Thời hạn {deadline} là bao lâu?",
    "Lịch {schedule} như thế nào?",
    "Thời gian {schedule} là khi nào?",

    # Problem-solving
    "Tôi gặp vấn đề {problem}, phải làm sao?",
    "Nếu {situation} thì phải xử lý thế nào?",
    "Trường hợp {situation} thì làm gì?",

    # Multi-turn starters
    "Tôi muốn hỏi về {topic}",
    "Bạn có thể giải thích {topic} không?",

    # Comparison
    "Sự khác nhau giữa {concept1} và {concept2} là gì?",
]


def fill_template(template: str, topic: str) -> str:
    """Fill all placeholder slots with the same topic string."""
    return template.format(
        topic=topic,
        action=topic,
        process=topic,
        procedure=topic,
        requirement=topic,
        event=topic,
        deadline=topic,
        schedule=topic,
        problem=topic,
        situation=topic,
        concept1=topic,
        concept2=topic,
    )


def create_chatml_pair(
    question: str,
    answer: str,
    system_prompt: str = SYSTEM_PROMPT_VI,
) -> dict:
    """Build a single ChatML-formatted training pair."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def create_multiturn_pair(
    turns: list[tuple[str, str]],
    system_prompt: str = SYSTEM_PROMPT_VI,
) -> dict:
    """Build a multi-turn ChatML conversation.

    Args:
        turns: List of (user_message, assistant_response) tuples.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in turns:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    return {"messages": messages}
