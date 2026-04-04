"""
Language-specific prompts for dataset generation.
Supports Vietnamese and English with customizable system and user prompts.
"""

SYSTEM_PROMPTS = {
    "english": """You are a helpful assistant that generates realistic text examples for a classification dataset.
Always respond in valid JSON format with exactly two fields: \"text\" and \"label\".
The \"text\" should be natural, diverse, and realistic.
The \"label\" must be one of the provided categories.
Do not add any extra explanation or markdown.""",

    "vietnamese": """Bạn là một trợ lý hỗ trợ tạo ra các ví dụ văn bản thực tế cho bộ dữ liệu phân loại.
Luôn trả lời dưới định dạng JSON hợp lệ với chính xác hai trường: \"text\" và \"label\".
\"text\" nên tự nhiên, đa dạng và thực tế.
\"label\" phải là một trong các danh mục được cung cấp.
Không thêm bất kỳ giải thích hay markdown nào."""
}

USER_PROMPTS = {
    "english": {
        "template": """Generate one realistic text example for {labels} classification in the domain of \"{domain}\".
The text should be {min_sentences}-{max_sentences} sentences long.
Respond ONLY with a JSON object like this:
{{
  \"text\": \"Your generated text here.\",
  \"label\": \"{first_label}\"
}}"""
    },

    "vietnamese": {
        "template": """Tạo một ví dụ văn bản thực tế cho bài toán phân loại {labels} trong lĩnh vực \"{domain}\".
Văn bản nên dài {min_sentences}-{max_sentences} câu.
Chỉ trả lời bằng một đối tượng JSON như thế này:
{{
  \"text\": \"Văn bản được tạo ra của bạn ở đây.\",
  \"label\": \"{first_label}\"
}}"""
    }
}

DOMAIN_TRANSLATIONS = {
    "english": {
        "general customer reviews": "general customer reviews",
        "product feedback": "product feedback",
        "social media": "social media",
        "news articles": "news articles",
        "emails": "emails",
        "tweets": "tweets",
        "restaurant reviews": "restaurant reviews",
        "movie reviews": "movie reviews",
    },
    "vietnamese": {
        "general customer reviews": "đánh giá khách hàng chung",
        "product feedback": "phản hồi sản phẩm",
        "social media": "mạng xã hội",
        "news articles": "bài báo tin tức",
        "emails": "emails",
        "tweets": "tweets",
        "restaurant reviews": "đánh giá nhà hàng",
        "movie reviews": "đánh giá phim ảnh",
    }
}

LANGUAGE_CHOICES = ["english", "vietnamese"]
LANGUAGE_DISPLAY = {
    "english": "🇬🇧 English",
    "vietnamese": "🇻🇳 Tiếng Việt"
}


def get_system_prompt(language: str = "english") -> str:
    language = language.lower()
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["english"])


def get_user_prompt(
    language: str = "english",
    labels: list = None,
    domain: str = "general customer reviews",
    min_sentences: int = 1,
    max_sentences: int = 3,
) -> str:
    if labels is None:
        labels = ["positive", "negative"]

    language = language.lower()
    template = USER_PROMPTS.get(language, USER_PROMPTS["english"])["template"]
    domain_translated = DOMAIN_TRANSLATIONS.get(language, {}).get(domain, domain)
    labels_str = "/".join(labels)
    first_label = labels[0] if labels else "positive"

    return template.format(
        labels=labels_str,
        domain=domain_translated,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        first_label=first_label,
    )
