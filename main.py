#!/usr/bin/env python
"""
命令行问答。

    python main.py                      # 交互模式
    python main.py "五粮液2024年营收多少"   # 单次提问
"""
import sys

from rag import app, cite


def ask(question: str, chat_history: list[str]) -> str:
    result = app.invoke({"question": question, "chat_history": chat_history})

    print("\n" + "=" * 60)
    print(result["generation"])
    sources = cite(result.get("documents", []))
    if sources:
        print("\n📚 依据:")
        print("\n".join(f"   {s}" for s in sources))
    print("=" * 60 + "\n")
    return result["generation"]


def main():
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), [])
        return

    print("🤖 年报问答已启动，输入问题开始，Ctrl-C 退出。\n")
    history: list[str] = []
    while True:
        try:
            question = input("❓ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not question:
            continue
        answer = ask(question, history)
        history.extend([f"用户: {question}", f"助手: {answer}"])
        history = history[-6:]   # 只留最近三轮，避免重写时被旧话题带偏


if __name__ == "__main__":
    main()
