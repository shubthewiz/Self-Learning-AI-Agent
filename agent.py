import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from groq import Groq
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.theme import Theme


load_dotenv()

MEMORY_FILE = Path("memory.json")
OUTPUT_DIR = Path("outputs")
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY_ITEMS = 8

custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
    }
)
console = Console(theme=custom_theme) 


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def slugify(text):
    return re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-") or "task"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_memory(history):
    try:
        MEMORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    except Exception as exc:
        console.print(f"[warning]Could not save memory:[/warning] {exc}")


def append_memory(entry):
    history = load_memory()
    history.append(entry)
    save_memory(history)


def history_for_task(task, limit=MAX_HISTORY_ITEMS):
    matching = [item for item in load_memory() if item.get("task") == task]
    return matching[-limit:]


def ensure_memory_file():
    if not MEMORY_FILE.exists():
        save_memory([])


def summarize_history(history):
    if not history:
        return "No previous attempts."

    lines = []
    for item in history[-MAX_HISTORY_ITEMS:]:
        status = item.get("status", "unknown")
        platform = item.get("platform", "unknown")
        error = item.get("error") or item.get("summary", "No details.")
        lines.append(f"Attempt {item.get('iteration', '?')} on {platform}: {status} - {error}")
    return "\n".join(lines)


def heuristic_plan(task, history):
    failed_platforms = {
        item.get("platform", "").lower()
        for item in history
        if item.get("status") == "failed"
    }

    if "linkedin" not in failed_platforms:
        platform = "linkedin"
        thought = "There is no failed LinkedIn attempt yet, so starting there gives the agent a first path to try."
    elif "indeed" not in failed_platforms:
        platform = "indeed"
        thought = "LinkedIn already failed, so switching to Indeed is the safest next move."
    elif "naukri" not in failed_platforms:
        platform = "naukri"
        thought = "LinkedIn and Indeed already failed, so trying Naukri gives the agent another structured jobs board."
    else:
        platform = "remoteok"
        thought = "The broader job sites already failed, so moving to RemoteOK is the best fallback."

    return {
        "thought": thought,
        "target_platform": platform,
        "search_query": task,
    }


def planner(task, history):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        console.print("[error]GROQ_API_KEY is missing. Using fallback planning logic.[/error]")
        return heuristic_plan(task, history)

    client = Groq(api_key=api_key)
    history_text = summarize_history(history)

    system_prompt = f"""
You are the planning brain for a browser agent. Your job is to choose the next website
and search query based on the current task and what previously failed.

Task:
{task}

Recent attempt history:
{history_text}

Rules:
- Prefer LinkedIn first only if it has not already failed for this task.
- If LinkedIn hit a login wall, CAPTCHA, timeout, or inaccessible page, pivot to Indeed.
- If Indeed is blocked or returns no extractable jobs, pivot to Naukri.
- If Naukri is blocked or returns no extractable jobs, pivot to RemoteOK.
- Avoid repeating the exact same failed platform unless every option has failed.
- Keep the search query concise and practical for a jobs website.

Respond with valid JSON only using exactly these keys:
- "thought": one sentence explaining the choice
- "target_platform": either "linkedin", "indeed", "naukri", or "remoteok"
- "search_query": concise query text for the jobs site
"""

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": system_prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        decision = json.loads(response.choices[0].message.content)
        target = decision.get("target_platform", "indeed").strip().lower()
        if target not in {"linkedin", "indeed", "naukri", "remoteok"}:
            target = heuristic_plan(task, history)["target_platform"]
        return {
            "thought": decision.get("thought", "Choosing the next platform based on prior failures."),
            "target_platform": target,
            "search_query": decision.get("search_query", task).strip() or task,
        }
    except Exception as exc:
        console.print(f"[warning]Planner fallback:[/warning] {exc}")
        return heuristic_plan(task, history)


def build_job_search_url(platform, query):
    encoded = quote_plus(query)
    if platform == "linkedin":
        return f"https://www.linkedin.com/jobs/search/?keywords={encoded}"
    if platform == "naukri":
        return f"https://www.naukri.com/{slugify(query)}-jobs?k={encoded}"
    if platform == "remoteok":
        return f"https://remoteok.com/remote-dev-jobs?location=worldwide&term={encoded}"
    return f"https://www.indeed.com/jobs?q={encoded}"


def make_absolute_url(base_url, href):
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("/"):
        if "linkedin.com" in base_url:
            return f"https://www.linkedin.com{href}"
        if "indeed.com" in base_url:
            return f"https://www.indeed.com{href}"
        if "naukri.com" in base_url:
            return f"https://www.naukri.com{href}"
        if "remoteok.com" in base_url:
            return f"https://remoteok.com{href}"
    return href


def first_text(locator, selectors):
    for selector in selectors:
        try:
            node = locator.locator(selector).first
            if node.count() == 0:
                continue
            text = " ".join(node.inner_text(timeout=2000).split()).strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def first_href(locator, selectors, base_url):
    for selector in selectors:
        try:
            node = locator.locator(selector).first
            if node.count() == 0:
                continue
            href = (node.get_attribute("href") or "").strip()
            if href:
                return make_absolute_url(base_url, href)
        except Exception:
            continue
    return ""


def detect_linkedin_block(page):
    body_text = page.locator("body").inner_text(timeout=5000).lower()
    block_markers = [
        "sign in",
        "join now",
        "captcha",
        "unusual activity",
        "login",
    ]
    return any(marker in body_text for marker in block_markers)


def detect_naukri_block(page):
    title_text = page.title().lower()
    body_text = page.locator("body").inner_text(timeout=5000).lower()
    block_markers = [
        "verify you are human",
        "access denied",
        "you don't have permission to access",
        "captcha",
        "forbidden",
        "temporarily unavailable",
        "errors.edgesuite.net",
    ]
    combined_text = f"{title_text} {body_text}"
    return any(marker in combined_text for marker in block_markers)


def extract_indeed_jobs(page, limit=5):
    page.wait_for_selector("div.job_seen_beacon, div.cardOutline", timeout=15000)
    cards = page.locator("div.job_seen_beacon, div.cardOutline")
    jobs = []

    for index in range(min(cards.count(), limit * 3)):
        card = cards.nth(index)
        title = first_text(card, ["a.jcs-JobTitle span", "a[data-testid='job-title'] span", "h2.jobTitle"])
        company = first_text(card, ["span[data-testid='company-name']", "[data-testid='company-name']", ".companyName"])
        location = first_text(card, ["div[data-testid='text-location']", ".companyLocation"])
        link = first_href(card, ["a.jcs-JobTitle", "a[data-testid='job-title']"], page.url)

        if not title:
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "link": link,
                "source": "indeed",
                "summary": "",
            }
        )
        if len(jobs) >= limit:
            break

    return jobs


def extract_linkedin_jobs(page, limit=5):
    page.wait_for_selector(".base-search-card, .job-search-card", timeout=12000)
    cards = page.locator(".base-search-card, .job-search-card")
    jobs = []

    for index in range(min(cards.count(), limit * 3)):
        card = cards.nth(index)
        title = first_text(card, [".base-search-card__title", ".job-search-card__title", "h3"])
        company = first_text(card, [".base-search-card__subtitle", "h4", ".artdeco-entity-lockup__subtitle"])
        location = first_text(card, [".job-search-card__location", ".base-search-card__metadata"])
        link = first_href(card, ["a.base-card__full-link", "a"], page.url)

        if not title:
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "link": link,
                "source": "linkedin",
                "summary": "",
            }
        )
        if len(jobs) >= limit:
            break

    return jobs


def extract_naukri_jobs(page, limit=5):
    page.wait_for_selector(".srp-jobtuple-wrapper, .jobTuple, article.jobTuple", timeout=15000)
    cards = page.locator(".srp-jobtuple-wrapper, .jobTuple, article.jobTuple")
    jobs = []

    for index in range(min(cards.count(), limit * 4)):
        card = cards.nth(index)
        title = first_text(card, ["a.title", ".row1 a.title", "h2 a", "a[data-job-id]"])
        company = first_text(card, [".comp-name", "a.comp-name", ".row2 span a", ".companyInfo a"])
        location = first_text(card, [".locWdth", ".loc-wrap .locWdth", ".row3 .locWdth", "[title*='location']"])
        link = first_href(card, ["a.title", ".row1 a.title", "h2 a", "a[data-job-id]"], page.url)

        if not title:
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location,
                "link": link,
                "source": "naukri",
                "summary": first_text(card, [".job-desc", ".job-description", ".tags-gt"]),
            }
        )
        if len(jobs) >= limit:
            break

    return jobs


def extract_remoteok_jobs(page, limit=5):
    page.wait_for_selector("tr.job", timeout=12000)
    cards = page.locator("tr.job")
    jobs = []

    for index in range(min(cards.count(), limit * 4)):
        card = cards.nth(index)
        title = first_text(card, ["h2", ".company_and_position [itemprop='title']"])
        company = first_text(card, ["h3", ".companyLink h3"])
        location = first_text(card, [".location", ".tags .location"])
        link = first_href(card, ["a.preventLink", "td.company.position.company_and_position a"], page.url)

        if not title or title.lower() == "apply":
            continue

        jobs.append(
            {
                "title": title,
                "company": company,
                "location": location or "Remote",
                "link": link,
                "source": "remoteok",
                "summary": "",
            }
        )
        if len(jobs) >= limit:
            break

    return jobs


def extract_job_summary(context, job, timeout=15000):
    if not job.get("link"):
        return ""

    detail_page = context.new_page()
    try:
        detail_page.goto(job["link"], wait_until="domcontentloaded", timeout=timeout)
        detail_page.wait_for_timeout(1200)
        body_text = " ".join(detail_page.locator("body").inner_text(timeout=5000).split())
        return body_text[:500]
    except Exception:
        return ""
    finally:
        detail_page.close()


def normalize_job(job):
    cleaned = dict(job)
    for key in ["title", "company", "location", "link", "summary", "source"]:
        cleaned[key] = " ".join(str(cleaned.get(key, "")).split()).strip()
    return cleaned


def score_job(job, query):
    query_terms = [term for term in re.split(r"\W+", query.lower()) if len(term) > 2]
    haystack = " ".join(
        [
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            job.get("summary", ""),
        ]
    ).lower()

    score = 0
    for term in query_terms:
        if term in haystack:
            score += 2
        if term in job.get("title", "").lower():
            score += 3
    if "remote" in haystack:
        score += 1
    return score


def rank_jobs(jobs, query):
    ranked = []
    for job in jobs:
        enriched = dict(job)
        enriched["score"] = score_job(enriched, query)
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def dedupe_jobs(jobs):
    seen = set()
    unique_jobs = []

    for job in jobs:
        job = normalize_job(job)
        key = (
            job.get("title", "").strip().lower(),
            job.get("company", "").strip().lower(),
            job.get("location", "").strip().lower(),
            job.get("link", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_jobs.append(job)

    return unique_jobs


def fill_job_limit(jobs, limit):
    if len(jobs) >= limit:
        return jobs[:limit]
    return jobs


def export_results(task, jobs):
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = slugify(task)
    json_path = OUTPUT_DIR / f"{slug}-{timestamp}.json"
    csv_path = OUTPUT_DIR / f"{slug}-{timestamp}.csv"

    try:
        json_path.write_text(json.dumps(jobs, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["score", "title", "company", "location", "source", "link", "summary"],
            )
            writer.writeheader()
            writer.writerows(jobs)
    except Exception as exc:
        console.print(f"[warning]Could not export results:[/warning] {exc}")
        return None, None

    return json_path, csv_path


def execute_on_platform(platform, query, headed=True, slow_mo=1200, keep_open=False):
    search_url = build_job_search_url(platform, query)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=not headed,
                args=["--disable-blink-features=AutomationControlled"],
                slow_mo=slow_mo if headed else 0,
            )
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            try:
                page.goto(search_url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(2500)

                if platform == "linkedin":
                    if detect_linkedin_block(page):
                        return {
                            "status": "failed",
                            "platform": platform,
                            "error": "LinkedIn login wall or access block detected.",
                            "url": page.url,
                            "jobs": [],
                        }

                    jobs = extract_linkedin_jobs(page)
                elif platform == "naukri":
                    if detect_naukri_block(page):
                        return {
                            "status": "failed",
                            "platform": platform,
                            "error": "Naukri bot check or access block detected.",
                            "url": page.url,
                            "jobs": [],
                        }

                    jobs = extract_naukri_jobs(page)
                elif platform == "remoteok":
                    jobs = extract_remoteok_jobs(page)
                else:
                    title = page.title().lower()
                    if "blocked" in title:
                        return {
                            "status": "failed",
                            "platform": platform,
                            "error": "Indeed returned a block page.",
                            "url": page.url,
                            "jobs": [],
                        }
                    jobs = extract_indeed_jobs(page)

                if not jobs:
                    return {
                        "status": "failed",
                        "platform": platform,
                        "error": f"No jobs extracted from {platform}.",
                        "url": page.url,
                        "jobs": [],
                    }

                enriched_jobs = []
                for job in jobs[:5]:
                    enriched_job = dict(job)
                    enriched_job["summary"] = extract_job_summary(context, enriched_job)
                    enriched_jobs.append(enriched_job)

                ranked_jobs = rank_jobs(fill_job_limit(dedupe_jobs(enriched_jobs), 5), query)
                json_path, csv_path = export_results(query, ranked_jobs)

                return {
                    "status": "success",
                    "platform": platform,
                    "summary": f"Extracted {len(ranked_jobs)} jobs from {platform}.",
                    "url": page.url,
                    "jobs": ranked_jobs,
                    "json_path": str(json_path),
                    "csv_path": str(csv_path),
                }
            except PlaywrightTimeoutError as exc:
                return {
                    "status": "failed",
                    "platform": platform,
                    "error": f"Timeout while loading or parsing the page: {exc}",
                    "url": page.url if page else search_url,
                    "jobs": [],
                }
            except Exception as exc:
                return {
                    "status": "failed",
                    "platform": platform,
                    "error": f"Execution error: {exc}",
                    "url": page.url if page else search_url,
                    "jobs": [],
                }
            finally:
                if headed and keep_open and result_is_terminal_platform(platform):
                    console.print("[info]Browser is open. Press Enter here to close it.[/info]")
                    try:
                        input()
                    except EOFError:
                        page.wait_for_timeout(10000)
                try:
                    context.close()
                except Exception:
                    pass
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as exc:
        return {
            "status": "failed",
            "platform": platform,
            "error": f"Browser startup failed: {exc}",
            "url": search_url,
            "jobs": [],
        }


def result_is_terminal_platform(platform):
    return platform in {"indeed", "remoteok", "linkedin"}


def display_result(result):
    if result["status"] != "success":
        console.print(f"[error]Attempt failed:[/error] {result['error']}")
        return

    table = Table(title="Jobs Found")
    table.add_column("#", style="info", justify="right")
    table.add_column("Title", style="success")
    table.add_column("Company")
    table.add_column("Location")
    table.add_column("Score", justify="right")

    for index, job in enumerate(result["jobs"], start=1):
        table.add_row(
            str(index),
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            str(job.get("score", 0)),
        )

    console.print(table)
    console.print(f"[success]Source:[/success] {result['url']}")
    if result.get("json_path"):
        console.print(f"[success]JSON:[/success] {result['json_path']}")
    if result.get("csv_path"):
        console.print(f"[success]CSV:[/success] {result['csv_path']}")


def run_agent(task, max_attempts=3, headed=True, slow_mo=1200, keep_open=False):
    ensure_memory_file()
    prior_history = history_for_task(task)

    for iteration in range(1, max_attempts + 1):
        console.print(Rule(f"Iteration {iteration}"))

        relevant_history = history_for_task(task)
        decision = planner(task, relevant_history)
        platform = decision["target_platform"]
        query = decision["search_query"]

        console.print(
            Panel(
                f"[bold yellow]Thought:[/bold yellow] {decision['thought']}\n"
                f"[bold yellow]Platform:[/bold yellow] {platform}\n"
                f"[bold yellow]Query:[/bold yellow] {query}",
                title="Agent Plan",
                border_style="yellow",
            )
        )

        try:
            with console.status(f"[info]Opening {platform} and searching the web...[/info]"):
                result = execute_on_platform(
                    platform,
                    query,
                    headed=headed,
                    slow_mo=slow_mo,
                    keep_open=keep_open and iteration == max_attempts,
                )
        except Exception as exc:
            result = {
                "status": "failed",
                "platform": platform,
                "error": f"Unexpected agent error: {exc}",
                "url": "",
                "jobs": [],
            }

        memory_entry = {
            "task": task,
            "iteration": len(prior_history) + iteration,
            "timestamp": utc_now(),
            "platform": platform,
            "query": query,
            **result,
        }
        append_memory(memory_entry)

        if result["status"] == "success":
            display_result(result)
            return result

        console.print(f"[warning]Learning from failure and retrying.[/warning]")

    return {
        "status": "failed",
        "platform": "none",
        "error": f"All {max_attempts} attempts failed for task: {task}",
        "jobs": [],
    }


def prompt_for_task():
    try:
        response = console.input("[bold cyan]What should I search for on the web? [/bold cyan]").strip()
    except EOFError:
        return ""
    return response


def build_task_from_input(user_input):
    cleaned = user_input.strip()
    if not cleaned:
        raise ValueError("A search request is required.")
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Interactive web agent with memory.")
    parser.add_argument("task", nargs="*", help="Task to run, for example: python developer jobs")
    parser.add_argument("--headless", action="store_true", help="Run without opening the browser window.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum retry attempts.")
    parser.add_argument("--clear-memory", action="store_true", help="Clear saved memory before running.")
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=1200,
        help="Delay visible browser actions in milliseconds. Increase this for slower, easier-to-follow recordings.",
    )
    parser.add_argument("--keep-open", action="store_true", help="Keep the browser open until you press Enter.")
    args = parser.parse_args()

    if args.clear_memory:
        save_memory([])
    else:
        ensure_memory_file()

    user_input = " ".join(args.task).strip() or prompt_for_task()
    if not user_input:
        raise SystemExit("No task provided. Pass a task or run interactively.")

    task = build_task_from_input(user_input)

    console.print(
        Panel.fit(
            "[bold green]PERSONAL AGENT READY[/bold green]",
            subtitle="Input -> Plan -> Web Action -> Learn",
            border_style="green",
        )
    )

    result = run_agent(
        task,
        max_attempts=max(args.max_attempts, 1),
        headed=not args.headless,
        slow_mo=max(args.slow_mo, 0),
        keep_open=args.keep_open,
    )
    if result["status"] != "success":
        console.print(f"[error]{result['error']}[/error]")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[warning]Run stopped by user.[/warning]")
        raise SystemExit(130)
    except Exception as exc:
        console.print(f"[error]Fatal error:[/error] {exc}")
        raise SystemExit(1)
