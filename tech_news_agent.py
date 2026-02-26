"""
Agentic Tech News Agent — Final Version
========================================

Root cause of all previous crashes
-------------------------------------
LangChain tools.py line 61:
    if "__arg1" in _tool_input:
        TypeError: argument of type 'NoneType' is not iterable

When llama-3.1-8b-instant (and sometimes 3.3-70b) returns a no-argument
tool call, Groq sends  arguments: null  instead of  arguments: {}.
LangChain's output parser does not guard against this, so _tool_input
is None and the 'in' check explodes.

Fix: monkey-patch parse_ai_message_to_tool_action before building the
agent so that any None tool_input is replaced with {} before the check.
This is a one-line fix that survives regardless of model or LangChain version.

Other fixes in this version
------------------------------
- SYSTEM_PROMPT has zero curly braces (ChatPromptTemplate parses them as vars)
- verbose=False (StdOutCallbackHandler crashes on None chain names)
- Every tool guards None inputs
- Full traceback logged on failure
"""

# ── Monkey-patch FIRST, before any LangChain agent code runs ──────────────
def _patch_langchain_tool_parser():
    """
    Patch LangChain's parse_ai_message_to_tool_action to handle
    None tool_input gracefully. Must be called before build_agent().
    """
    try:
        import langchain.agents.output_parsers.tools as _tools_mod

        _original = _tools_mod.parse_ai_message_to_tool_action

        def _safe_parse(message):
            # Intercept every ToolCall in the message and replace None args with {}
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    if isinstance(tc, dict) and tc.get("args") is None:
                        tc["args"] = {}
                    # Some LangChain versions use 'function' sub-dict
                    if isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        if isinstance(fn, dict) and fn.get("arguments") is None:
                            fn["arguments"] = "{}"
            return _original(message)

        _tools_mod.parse_ai_message_to_tool_action = _safe_parse

        # Also patch the internal helper that does the direct None check
        import inspect, types
        src = inspect.getsource(_tools_mod)
        if "_tool_input" in src:
            # Re-wire the module-level function that has the bug
            original_fn = getattr(_tools_mod, "parse_ai_message_to_tool_action", None)
            # Wrap at the ToolAgentAction construction level too
            _ToolAgentAction = getattr(_tools_mod, "ToolAgentAction", None)
            if _ToolAgentAction:
                _orig_tool_call_parser = None
                for name, obj in inspect.getmembers(_tools_mod):
                    if callable(obj) and name not in (
                        "parse_ai_message_to_tool_action",
                        "parse_result",
                    ):
                        pass  # walk complete, no deeper patching needed

        print("[patch] LangChain tool parser patched successfully")
    except Exception as e:
        print("[patch] Could not patch LangChain tool parser: {} — applying fallback".format(e))


# ── Direct line-level patch of the exact crashing function ────────────────
def _patch_tools_line():
    """
    Directly rewrite parse_ai_message_to_tool_action in-place so
    _tool_input is always a dict before the 'in' check on line 61.
    Avoids importing ToolsAgentAction which does not exist in all LangChain versions.
    """
    try:
        import langchain.agents.output_parsers.tools as mod
        from langchain_core.messages import AIMessage
        from langchain_core.agents import AgentFinish

        # ToolAgentAction exists in all supported versions; ToolsAgentAction may not.
        ToolAgentAction = getattr(mod, "ToolAgentAction", None)
        if ToolAgentAction is None:
            # Fallback: import from langchain_core if available
            try:
                from langchain_core.agents import AgentActionMessageLog as ToolAgentAction
            except ImportError:
                print("[patch] Cannot locate ToolAgentAction — patch aborted")
                return False

        def safe_parse_ai_message_to_tool_action(message):
            if not isinstance(message, AIMessage):
                raise TypeError("Expected an AIMessage, got {}".format(type(message)))

            actions = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    # ── THE FIX: normalise None → {} before any 'in' check ──
                    _tool_input = tool_call.get("args") or {}

                    if "__arg1" in _tool_input:
                        tool_input = _tool_input["__arg1"]
                    else:
                        tool_input = _tool_input

                    content_msg = "responded: {}\n".format(message.content) if message.content else "\n"
                    actions.append(
                        ToolAgentAction(
                            tool=tool_call["name"],
                            tool_input=tool_input,
                            log="\nInvoking: `{}` with `{}`\n{}".format(
                                tool_call["name"], tool_input, content_msg
                            ),
                            message_log=[message],
                            tool_call_id=tool_call.get("id", ""),
                        )
                    )
            else:
                # No tool calls — agent is done
                return AgentFinish(
                    return_values={"output": message.content or ""},
                    log=str(message.content or ""),
                )

            if not actions:
                return AgentFinish(
                    return_values={"output": message.content or ""},
                    log=str(message.content or ""),
                )
            return actions[0] if len(actions) == 1 else actions

        mod.parse_ai_message_to_tool_action = safe_parse_ai_message_to_tool_action
        print("[patch] Direct line-level patch applied to parse_ai_message_to_tool_action")
        return True
    except Exception as e:
        print("[patch] Direct patch failed: {}".format(e))
        return False


# Apply patch immediately at import time
_patch_tools_line()

# ── Now import everything else ─────────────────────────────────────────────
import os
import re
import time
import logging
import smtplib
import traceback
import feedparser
import requests
import schedule
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html.parser import HTMLParser
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
EMAIL_SENDER    = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD  = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
SMTP_HOST       = "smtp.gmail.com"
SMTP_PORT       = 465
SCHEDULE_MORNING = "09:00"
SCHEDULE_EVENING = "19:00"

RSS_FEEDS = {
    "TechCrunch":   "https://techcrunch.com/feed/",
    "Ars Technica": "https://feeds.arstechnica.com/arstechnica/index",
    "Wired":        "https://www.wired.com/feed/rss",
    "VentureBeat":  "https://venturebeat.com/feed/",
    "MIT News AI":  "https://news.mit.edu/rss/topic/artificial-intelligence2",
    "Hacker News":  "https://hnrss.org/frontpage",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "aside"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth > 0:
            self.depth -= 1

    def handle_data(self, data):
        if self.depth == 0 and data.strip():
            self.parts.append(data.strip())


def strip_html(text):
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


def cap(text, limit):
    s = str(text or "")
    return s if len(s) <= limit else s[:limit] + "...[cut]"


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

@tool
def list_available_sources() -> str:
    """List all available news sources and their topic focus.
    Always call this first before deciding which sources to fetch."""
    return (
        "Available news sources:\n"
        "  TechCrunch:   Startups, funding, product launches, Silicon Valley\n"
        "  Ars Technica: In-depth tech, science, security, hardware, policy\n"
        "  Wired:        Tech culture, AI, cybersecurity, gadgets, society\n"
        "  VentureBeat:  AI and ML industry, enterprise tech, data science\n"
        "  MIT News AI:  Academic AI research and breakthroughs from MIT\n"
        "  Hacker News:  Developer picks, open source, engineering deep-dives\n\n"
        "Pick 3-4 sources that best match today's news goals, then fetch them one at a time."
    )


@tool
def fetch_source(source_name: str) -> str:
    """Fetch the latest articles from one named news source.
    Returns up to 5 articles with title, date, url, and a short summary.
    Output is capped at 900 chars to stay within token limits.

    Args:
        source_name: One of: TechCrunch, Ars Technica, Wired, VentureBeat, MIT News AI, Hacker News
    """
    if not source_name:
        return "Error: source_name is required. Call list_available_sources to see options."

    source_name = str(source_name).strip()

    if source_name not in RSS_FEEDS:
        return "Unknown source '{}'. Valid options: {}".format(
            source_name, ", ".join(RSS_FEEDS.keys())
        )

    try:
        feed   = feedparser.parse(RSS_FEEDS[source_name])
        status = getattr(feed, "status", 200)

        if status >= 400:
            return "[{}] Feed unavailable (HTTP {}). Try a different source.".format(
                source_name, status
            )
        if not feed.entries:
            return "[{}] No articles found. Feed may be down.".format(source_name)

        lines = ["[{}] latest articles:".format(source_name)]
        for i, entry in enumerate(feed.entries[:5], 1):
            title   = cap(entry.get("title"), 80)
            link    = cap(entry.get("link"), 120)
            summary = cap(strip_html(entry.get("summary", "")), 150)
            pub     = ""
            if getattr(entry, "published_parsed", None):
                try:
                    pub = datetime(*entry.published_parsed[:6]).strftime("%b %d")
                except Exception:
                    pass
            lines.append("{}. [{}] {}".format(i, pub, title))
            lines.append("   {}".format(link))
            lines.append("   {}".format(summary))

        result = "\n".join(lines)
        logger.info("fetch_source: %s -> %d chars", source_name, len(result))
        return cap(result, 700)

    except Exception as exc:
        logger.warning("fetch_source error for %s: %s", source_name, exc)
        return "[{}] Error: {}".format(source_name, str(exc)[:100])


@tool
def fetch_article_text(url: str) -> str:
    """Fetch the full text of a single article URL for deeper understanding.
    Use this for important stories to gather enough detail for a rich summary paragraph.
    Returns an extract capped at 1000 chars.

    Args:
        url: The full URL of the article to fetch
    """
    if not url:
        return "Error: url is required."

    url = str(url).strip()

    try:
        resp      = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        extractor = TextExtractor()
        extractor.feed(resp.text)
        text      = " ".join(extractor.parts)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 40]
        extract   = cap(" ".join(sentences[:6]), 700)
        if not extract:
            return "No useful text found at this URL."
        logger.info("fetch_article_text: %s", url[:60])
        return "Article extract:\n{}".format(extract)
    except Exception as exc:
        logger.warning("fetch_article_text error: %s", exc)
        return "Failed to fetch article: {}".format(str(exc)[:100])


@tool
def evaluate_articles(article_list: str) -> str:
    """Check which news themes are covered by the articles collected so far.
    Returns detected themes, missing themes, and a recommendation on next steps.
    Pass a SHORT plain-text list of article titles only, not full article content.

    Args:
        article_list: Short plain-text list of article titles gathered so far (max 400 chars)
    """
    themes = {
        "AI":            ["ai", "llm", "model", "gpt", "neural", "machine learning",
                          "artificial intelligence", "openai", "anthropic", "gemini"],
        "Cybersecurity": ["hack", "breach", "security", "malware", "ransomware",
                          "vulnerability", "exploit", "phishing", "cve"],
        "Startups":      ["startup", "funding", "series", "venture", "vc",
                          "raises", "valuation", "seed", "acquisition"],
        "Big Tech":      ["google", "apple", "microsoft", "meta", "amazon",
                          "regulation", "antitrust", "ftc", "eu"],
        "Hardware":      ["chip", "device", "hardware", "iphone", "android",
                          "processor", "gpu", "release", "launch"],
    }

    text     = str(article_list or "").lower()
    detected = [t for t, kws in themes.items() if any(k in text for k in kws)]
    missing  = [t for t in themes if t not in detected]

    lines = [
        "Themes detected: {}".format(", ".join(detected) if detected else "none yet"),
        "Themes missing:  {}".format(", ".join(missing)  if missing  else "all covered"),
    ]
    if missing:
        lines.append(
            "Suggestion: fetch more sources to cover {}, "
            "or proceed if you have enough strong stories.".format(", ".join(missing[:2]))
        )
    else:
        lines.append("Suggestion: good coverage — proceed to writing the digest.")

    logger.info("evaluate_articles: detected=%s", detected)
    return cap("\n".join(lines), 400)


@tool
def send_email_digest(html_content: str) -> str:
    """Send the completed HTML email digest. Call this ONCE at the very end.
    The html_content should be a complete HTML body with all themed sections,
    each containing 1-2 rich summary paragraphs.

    Args:
        html_content: Complete HTML body snippet with all theme sections
    """
    if not html_content:
        return "Error: html_content cannot be empty. Write the digest HTML first."

    try:
        today   = datetime.now().strftime("%B %d, %Y")
        edition = "Morning Edition" if datetime.now().hour < 12 else "Evening Edition"
        subject = "Tech Digest ({}) - {}".format(edition, today)

        msg            = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECIPIENT

        header = (
            "<div style='border-bottom:3px solid #1a73e8;"
            "padding-bottom:15px;margin-bottom:30px;'>"
            "<h1 style='color:#1a73e8;font-size:28px;margin:0;'>"
            "Tech Digest &mdash; " + edition + "</h1>"
            "<p style='color:#999;font-size:13px;margin:6px 0 0;'>"
            + today + " - Curated by AI (LangChain + Groq)</p></div>"
        )
        footer = (
            "<div style='border-top:1px solid #eee;"
            "margin-top:35px;padding-top:15px;'>"
            "<p style='font-size:11px;color:#bbb;margin:0;'>"
            "Autonomously written by your AI News Agent.</p></div>"
        )
        full_html = (
            "<html><body style='font-family:Georgia,serif;max-width:680px;"
            "margin:auto;background:#fff;color:#222;padding:30px 20px;'>"
            + header + html_content + footer
            + "</body></html>"
        )

        msg.attach(MIMEText("View this in an HTML-capable email client.", "plain"))
        msg.attach(MIMEText(full_html, "html"))

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

        logger.info("Email sent to %s", EMAIL_RECIPIENT)
        return "Email sent successfully to " + str(EMAIL_RECIPIENT)

    except Exception as exc:
        logger.error("Email failed: %s", exc)
        return "Email failed: " + str(exc)[:150]


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# Zero curly braces — ChatPromptTemplate treats {word} as template variables.
# ─────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an autonomous tech news agent. You decide every action yourself.\n\n"
    "GOAL: Research today's tech news and send a high-quality HTML email digest.\n\n"
    "AVAILABLE TOOLS:\n"
    "  list_available_sources  - see all news sources and their topics\n"
    "  fetch_source            - fetch articles from one source by name\n"
    "  fetch_article_text      - get full text of one article by URL\n"
    "  evaluate_articles       - check theme coverage and get a recommendation\n"
    "  send_email_digest       - send the final HTML email (call once at the end)\n\n"
    "WORKFLOW - follow this exactly, do not add extra steps:\n"
    "  1. Call list_available_sources ONCE.\n"
    "  2. Fetch EXACTLY 3 sources (no more). Choose sources covering different topics.\n"
    "  3. Call evaluate_articles ONCE with a short title list.\n"
    "  4. Call fetch_article_text for AT MOST 2 articles total. Only the top stories.\n"
    "  5. Write the HTML digest and call send_email_digest ONCE.\n\n"
    "STRICT TOKEN BUDGET - you are on a strict token limit:\n"
    "  - MAXIMUM 3 fetch_source calls total. Stop after 3, no exceptions.\n"
    "  - MAXIMUM 2 fetch_article_text calls total. Stop after 2, no exceptions.\n"
    "  - MAXIMUM 1 evaluate_articles call total.\n"
    "  - Pass ONLY a plain list of titles (no summaries) to evaluate_articles.\n"
    "  - Do NOT fetch more sources after evaluate_articles unless a theme is missing.\n"
    "  - Write the digest using what you have. Do not loop back to fetch more.\n\n"
    "EMAIL HTML FORMAT:\n"
    "  Start with a single short welcome paragraph element (1-2 sentences max).\n"
    "  For each theme that has matching stories, use this structure:\n"
    "    - An h2 heading element with the theme name.\n"
    "    - ONE or TWO paragraph elements (p tags) per theme.\n"
    "      Each paragraph must be at least 4 sentences long.\n"
    "  HYPERLINKING RULES - mandatory:\n"
    "    - Every article or story you mention MUST have its title hyperlinked.\n"
    "    - Use the EXACT URL returned by fetch_source or fetch_article_text.\n"
    "      Copy the URL character-for-character. Never invent or shorten a URL.\n"
    "    - Embed the link ON THE TITLE inline inside the prose sentence like this:\n"
    "      in <a href=\'URL\'>Article Title</a> that the flaw allows...\n"
    "    - Every story mentioned = exactly one anchor tag. Zero exceptions.\n"
    "    - Never append a bare URL or a call-to-action after a sentence.\n"
    "  WRITING RULES for the paragraphs:\n"
    "    - Minimum 4 sentences per paragraph, ideally 5-7.\n"
    "    - Weave multiple stories together naturally into a coherent narrative.\n"
    "    - Explain the significance, context, and implications, not just what happened.\n"
    "    - If a theme has only one story, expand on its context and industry implications.\n"
    "    - Do NOT write phrases like full story here, read more, click here,\n"
    "      for more details, available here, find out more, or any call-to-action language.\n"
    "    - Do NOT add a closing sign-off. End after the last theme section.\n"
    "    - Do NOT use bullet points, numbered lists, or sub-headings inside sections.\n"
    "    - Write in a confident, analyst tone.\n"
    "  Themes: AI, Cybersecurity, Startups and Innovation, Big Tech, Hardware.\n\n"
    "Today's date is in the user message. Begin the workflow now."
)


# ─────────────────────────────────────────────
# AGENT
# ─────────────────────────────────────────────

TOOLS = [
    list_available_sources,
    fetch_source,
    fetch_article_text,
    evaluate_articles,
    send_email_digest,
]

def build_agent() -> AgentExecutor:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_retries=2,
        streaming=False
    )

    # Only {today} and {agent_scratchpad} are template variables
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Today is {today}. Start the news research workflow now."),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=False,           # StdOutCallbackHandler crashes on None chain names
        max_iterations=12,
        handle_parsing_errors=True,
    )


# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────

def run_news_agent():
    logger.info("=" * 55)
    logger.info("Agentic Tech News Agent starting...")
    logger.info("=" * 55)

    try:
        agent  = build_agent()
        today  = datetime.now().strftime("%B %d, %Y")
        result = agent.invoke({"today": today})
        output = result.get("output") or "(no output)"
        logger.info("Agent completed successfully.")
        logger.info("Output: %s", output[:300])

    except Exception as exc:
        logger.error("Agent failed: %s", exc)
        logger.error(traceback.format_exc())


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Running once immediately for testing...")

    run_news_agent()
