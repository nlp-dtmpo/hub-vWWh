from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


PLACEHOLDER_API_KEY: str = "YOUR_API_KEY_HERE"
DEFAULT_API_KEY: str = "xx"
DEFAULT_BASE_URL: str = "https://api.deepseek.com/v1"
DEFAULT_MODEL: str = "deepseek-chat"


@dataclass(frozen=True)
class AgentResponse:
    """Structured response for one question."""

    question: str
    sql: str
    rows: List[Tuple[Any, ...]]
    answer: str


class ChinookNL2SQLAgent:
    """
    Real NL2SQL agent for sqlite(chinook.db).

    Workflow:
    1) Read schema.
    2) Generate SQL via LLM.
    3) Execute SQL (read-only).
    4) Retry with DB error feedback.
    5) Summarize rows as Chinese answer.
    """

    def __init__(
        self,
        db_path: Path,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        max_retry: int = 3,
    ) -> None:
        self.db_path: Path = db_path
        self.api_key: str = api_key
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.max_retry: int = max_retry
        self._validate_db_path()

    def _validate_db_path(self) -> None:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        if not self.db_path.is_file():
            raise ValueError(f"Database path is not a file: {self.db_path}")

    def ask(self, question: str) -> AgentResponse:
        """Question -> SQL -> rows -> answer."""
        schema_text: str = self.get_schema_text()
        last_error: Optional[str] = None
        final_sql: str = ""
        final_rows: List[Tuple[Any, ...]] = []

        for _ in range(self.max_retry):
            final_sql = self.nl2sql(question=question, schema_text=schema_text, last_error=last_error)
            self.ensure_readonly_sql(final_sql)
            ok, rows, err = self.try_execute_sql(final_sql)
            if ok:
                final_rows = rows
                break
            last_error = err
        else:
            raise RuntimeError(f"SQL generation failed after retries. Last error: {last_error}")

        answer: str = self.summarize_answer(question=question, sql=final_sql, rows=final_rows)
        return AgentResponse(question=question, sql=final_sql, rows=final_rows, answer=answer)

    def get_schema_text(self) -> str:
        """Collect table/column schema from sqlite."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            table_names: List[str] = [x[0] for x in cursor.fetchall()]
            lines: List[str] = []
            for table_name in table_names:
                lines.append(f"Table: {table_name}")
                cursor.execute(f"PRAGMA table_info('{table_name}');")
                columns = cursor.fetchall()
                for _, col_name, col_type, not_null, _, pk in columns:
                    lines.append(
                        f"  - {col_name} {col_type} "
                        f"(not_null={bool(not_null)}, primary_key={bool(pk)})"
                    )
                lines.append("")
        return "\n".join(lines).strip()

    def nl2sql(self, question: str, schema_text: str, last_error: Optional[str]) -> str:
        """Generate SQL by LLM using schema + question + last error."""
        error_hint: str = last_error if last_error else "None"
        prompt: str = f"""
你是专业 SQLite 数据分析助手。把用户问题翻译为 SQL。

强约束:
1. 只输出 SQL，不要解释，不要 markdown。
2. 只能输出一条 SQL。
3. 只能使用 SELECT / WITH 语句，禁止增删改表。
4. SQL 必须兼容 SQLite。
5. 优先使用数据库中已存在的表和字段。

数据库 Schema:
{schema_text}

用户问题:
{question}

上一轮执行报错:
{error_hint}
""".strip()
        llm_text: str = self.call_llm(prompt)
        return self.extract_sql(llm_text)

    def summarize_answer(self, question: str, sql: str, rows: Sequence[Tuple[Any, ...]]) -> str:
        """Turn SQL results into concise Chinese answer via LLM."""
        prompt: str = f"""
你是数据问答助手。请基于问题、SQL与结果，输出简洁中文回答。
若结果为空，明确说明“未查到数据”。

问题:
{question}

SQL:
{sql}

结果:
{self.rows_to_text(rows)}
""".strip()
        return self.call_llm(prompt)

    def call_llm(self, user_prompt: str) -> str:
        """Call OpenAI-compatible API."""
        normalized_key: str = self.api_key.strip()
        if not normalized_key or normalized_key == PLACEHOLDER_API_KEY:
            raise ValueError(
                "API key is placeholder. Please replace DEFAULT_API_KEY "
                "or pass api_key when constructing ChinookNL2SQLAgent."
            )

        url: str = f"{self.base_url}/chat/completions"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.1,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        try:
            content: str = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected LLM response: {json.dumps(data, ensure_ascii=False)}") from exc

    @staticmethod
    def extract_sql(text: str) -> str:
        """Extract SQL from plain text or fenced code."""
        cleaned: str = text.strip()
        if "```" not in cleaned:
            return cleaned.rstrip(";") + ";"
        start = cleaned.find("```")
        end = cleaned.rfind("```")
        if start == end:
            return cleaned.rstrip(";") + ";"
        block = cleaned[start + 3 : end].strip()
        if block.lower().startswith("sql"):
            block = block[3:].strip()
        return block.rstrip(";") + ";"

    @staticmethod
    def ensure_readonly_sql(sql: str) -> None:
        """Protect local DB: only allow read-only SQL."""
        normalized: str = sql.strip().lower()
        if not normalized.startswith(("select", "with")):
            raise ValueError(f"Only SELECT/WITH SQL is allowed: {sql}")
        forbidden_tokens = (
            "insert ",
            "update ",
            "delete ",
            "drop ",
            "alter ",
            "create ",
            "replace ",
            "truncate ",
        )
        if any(token in normalized for token in forbidden_tokens):
            raise ValueError(f"Forbidden SQL detected: {sql}")

    def try_execute_sql(self, sql: str) -> Tuple[bool, List[Tuple[Any, ...]], Optional[str]]:
        """Attempt SQL execution and catch DB errors."""
        try:
            rows = self.execute_sql(sql)
            return True, rows, None
        except Exception as exc:
            return False, [], str(exc)

    def execute_sql(self, sql: str) -> List[Tuple[Any, ...]]:
        """Execute SQL and return fetched rows."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()

    @staticmethod
    def rows_to_text(rows: Sequence[Tuple[Any, ...]], max_rows: int = 20) -> str:
        """Convert rows to prompt-safe preview text."""
        if not rows:
            return "[]"
        preview: List[Tuple[Any, ...]] = list(rows[:max_rows])
        suffix: str = "" if len(rows) <= max_rows else f"\n... total_rows={len(rows)}"
        return f"{preview}{suffix}"


def get_default_db_path() -> Path:
    """Locate chinook.db relative to Week12 root."""
    current_file: Path = Path(__file__).resolve()
    week12_dir: Path = current_file.parent.parent
    return week12_dir / "04_SQL-Code-Agent-Demo" / "chinook.db"


def run_interactive_demo() -> None:
    """Interactive CLI for any database-related question."""
    agent = ChinookNL2SQLAgent(db_path=get_default_db_path())
    print("Chinook NL2SQL Agent 已启动。输入问题，输入 exit 结束。")
    while True:
        question: str = input("\n请输入问题: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break
        if not question:
            print("问题不能为空。")
            continue
        try:
            result = agent.ask(question)
            print(f"SQL: {result.sql}")
            print(f"结果: {result.rows}")
            print(f"回答: {result.answer}")
        except Exception as exc:
            print(f"执行失败: {exc}")


if __name__ == "__main__":
    run_interactive_demo()
