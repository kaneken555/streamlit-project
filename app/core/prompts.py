def build_system_prompt(base_system: str, context: str) -> str:
    return (
        (base_system or "")
        + "\n\n# 参考資料（抜粋）\n"
        + (context or "（該当資料なし）")
        + "\n\n※上記の資料のみを根拠に、日本語で簡潔に回答してください。"
        + "\n必要に応じて [番号] を使って根拠を示してください。"
    )
