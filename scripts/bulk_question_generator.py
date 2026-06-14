"""
Tamkeen Bulk Question Generator
================================
Scans all course transcripts from the Tamkeen Content project_data,
uses Gemini 2.5 Pro to generate 20 MCQs per course (4 per difficulty level),
and outputs Prisma-compatible JSON files.

Each question includes both a standard `question_text` and a cognitively
accessible `simplified_text` for students with disabilities (Dyslexia/ADHD).

Usage:
    python scripts/bulk_question_generator.py

Requirements:
    - GEMINI_API_KEY in .env file
    - Tamkeen Content at CONTENT_ROOT path
    - data/generated/assessment_mapping.json filled with real DB IDs
"""

import os
import sys
import io
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Tamkeen Content data source
CONTENT_ROOT = Path(r"F:\Cs\Projects\Tamkeen\Tamkeen Content\project_data\courses")

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"

# Generation settings
QUESTIONS_PER_COURSE = 20
DIFFICULTY_DISTRIBUTION = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4}
GEMINI_MODEL = "gemini-2.5-flash"
MAX_TRANSCRIPT_CHARS = 90_000  # Stay within context window safely

# Assessment mapping file
MAPPING_FILE = OUTPUT_DIR / "assessment_mapping.json"

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_generator")


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class LessonData:
    """Represents a single lesson's transcript data."""
    lesson_key: str
    title: str
    chapter: str
    transcript_ar: str
    transcript_en: str = ""


@dataclass
class CourseData:
    """Aggregated course data ready for question generation."""
    slug: str
    title: str
    num_lessons: int
    lessons: list[LessonData] = field(default_factory=list)
    chapters: list[str] = field(default_factory=list)

    @property
    def full_transcript(self) -> str:
        """Build chapter-aware concatenated transcript."""
        parts = []
        current_chapter = None
        for lesson in self.lessons:
            if lesson.chapter != current_chapter:
                current_chapter = lesson.chapter
                parts.append(f"\n{'='*60}")
                parts.append(f"=== {current_chapter} ===")
                parts.append(f"{'='*60}\n")
            parts.append(f"[Lesson: {lesson.title}]")
            parts.append(lesson.transcript_ar)
            if lesson.transcript_en:
                parts.append(f"[English supplement]")
                parts.append(lesson.transcript_en)
            parts.append("")
        return "\n".join(parts)

    @property
    def total_chars(self) -> int:
        return len(self.full_transcript)


@dataclass
class Question:
    """A single generated question."""
    question_text: str
    simplified_text: str
    question_type: str
    correct_answer: str
    options: list[str]
    difficulty_level: int


@dataclass
class ValidationResult:
    """Result from the LLM-as-a-Judge validation."""
    valid: bool
    flagged_indices: list[int] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    difficulty_distribution: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 1: Scan & Aggregate Transcripts
# ---------------------------------------------------------------------------

def scan_courses() -> list[CourseData]:
    """
    Scan all course folders in CONTENT_ROOT.
    Read course.json for lesson ordering, then load transcripts.
    """
    courses = []

    if not CONTENT_ROOT.exists():
        logger.error(f"Content root not found: {CONTENT_ROOT}")
        sys.exit(1)

    for course_dir in sorted(CONTENT_ROOT.iterdir()):
        if not course_dir.is_dir():
            continue

        course_json_path = course_dir / "meta" / "course.json"
        if not course_json_path.exists():
            logger.warning(f"No course.json found for {course_dir.name}, skipping.")
            continue

        with open(course_json_path, "r", encoding="utf-8") as f:
            course_meta = json.load(f)

        slug = course_meta["course_slug"]
        title = course_meta["title"]
        num_lessons = course_meta["number_of_lessons"]

        transcripts_dir = course_dir / "transcripts_unified"
        if not transcripts_dir.exists():
            logger.warning(f"No transcripts_unified/ for {slug}, skipping.")
            continue

        # Parse lessons in order, extract chapter info from lesson keys
        lessons = []
        chapters_seen = set()

        for lesson_meta in course_meta.get("lessons", []):
            lesson_key = lesson_meta["lesson_key"]
            lesson_title = lesson_meta["title"]

            # Extract chapter from title (e.g., "CH01_VID01_..." -> "Chapter 1")
            chapter = _extract_chapter(lesson_title, slug)

            # Skip welcome / intro lessons for question generation
            if _is_non_content_lesson(lesson_key):
                continue

            # Read Arabic transcript (primary)
            ar_path = transcripts_dir / f"{lesson_key}.ar.txt"
            transcript_ar = _read_file_safe(ar_path)

            if not transcript_ar:
                logger.debug(f"  No Arabic transcript for {lesson_key}")
                continue

            # Read English transcript (supplementary, if available)
            en_path = transcripts_dir / f"{lesson_key}.en.txt"
            transcript_en = _read_file_safe(en_path)

            lessons.append(LessonData(
                lesson_key=lesson_key,
                title=lesson_title,
                chapter=chapter,
                transcript_ar=transcript_ar,
                transcript_en=transcript_en,
            ))

            chapters_seen.add(chapter)

        if not lessons:
            logger.warning(f"No valid lessons found for {slug}, skipping.")
            continue

        course = CourseData(
            slug=slug,
            title=title,
            num_lessons=num_lessons,
            lessons=lessons,
            chapters=sorted(chapters_seen),
        )

        logger.info(
            f"Scanned: {slug} — {len(lessons)} lessons, "
            f"{len(chapters_seen)} chapters, {course.total_chars:,} chars"
        )
        courses.append(course)

    return courses


def _extract_chapter(lesson_title: str, course_slug: str) -> str:
    """Extract chapter identifier from lesson title."""
    title_upper = lesson_title.upper()
    # Match patterns like CH01, CH02, CHAPTER 1, etc.
    if "CH0" in title_upper or "CH1" in title_upper:
        for part in title_upper.replace("_", " ").split():
            if part.startswith("CH") and len(part) <= 5:
                try:
                    num = int(part[2:])
                    return f"Chapter {num}"
                except ValueError:
                    pass
    return "General"


def _is_non_content_lesson(lesson_key: str) -> bool:
    """Check if a lesson is non-content (welcome, intro, etc.)."""
    skip_patterns = ["welcome", "what_we_will_learn", "intro", "conclusion"]
    key_lower = lesson_key.lower()
    return any(pattern in key_lower for pattern in skip_patterns)


def _read_file_safe(filepath: Path) -> str:
    """Read file contents, return empty string on failure."""
    if not filepath.exists():
        return ""
    try:
        return filepath.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Failed to read {filepath.name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Phase 2: Generate Questions via Gemini
# ---------------------------------------------------------------------------

def _build_generation_prompt(course: CourseData) -> str:
    """Build the structured generation prompt for Gemini."""

    chapter_summary = "\n".join(
        f"  - {ch}" for ch in course.chapters
    )

    transcript = course.full_transcript

    # Truncate if too long, keeping chapter distribution fair
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        logger.warning(
            f"Transcript for {course.slug} is {len(transcript):,} chars, "
            f"truncating to {MAX_TRANSCRIPT_CHARS:,}"
        )
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n\n[... transcript truncated ...]"

    return f"""You are an expert educational assessment designer for the Tamkeen e-learning platform.

COURSE: {course.title}
CHAPTERS:
{chapter_summary}

TRANSCRIPT (Arabic with some English supplements):
---
{transcript}
---

TASK:
Generate exactly 20 multiple-choice questions in ENGLISH based ONLY on the transcript content above.

DISTRIBUTION (strictly enforced):
- Level 1 (Basic Recall): 4 questions — definitions, terminology, simple facts
- Level 2 (Understanding): 4 questions — explain concepts in own words
- Level 3 (Application): 4 questions — apply concepts to practical scenarios
- Level 4 (Analysis): 4 questions — compare, contrast, debug, identify relationships
- Level 5 (Evaluation): 4 questions — design decisions, trade-offs, best practices

FOR EACH QUESTION provide ALL of these fields:
1. "question_text": The standard question (clear, professional English)
2. "simplified_text": A cognitively accessible version for students with Dyslexia/ADHD:
   - Maximum 12 words
   - No jargon — replace technical terms with simple descriptions
   - Direct and unambiguous
3. "options": Array of exactly 4 string choices
4. "correct_answer": The letter "A", "B", "C", or "D" corresponding to the correct option
5. "difficulty_level": Integer 1 through 5
6. "question_type": Always "multiple_choice"

RULES:
- Every question MUST be answerable from the transcript content — do not invent facts
- DO NOT reference the transcript, video, or lesson in the question text (e.g., avoid "According to the transcript", "In the video", or "Based on the text"). Ask the question directly.
- Distractors must be plausible but clearly wrong
- No trick questions or "all of the above" / "none of the above" options
- Distribute correct answers roughly evenly across A, B, C, D
- simplified_text must genuinely simplify — not just shorten
- Cover material from ALL chapters, not just the first or last
- Questions within the same difficulty level should test DIFFERENT concepts

Return ONLY a valid JSON array of 20 objects. No markdown fences, no explanation, no extra text."""


GENERATION_PROMPT_TEMPLATE = _build_generation_prompt  # alias for clarity


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda rs: logger.warning(
        f"  Retrying generation (attempt {rs.attempt_number})..."
    ),
)
def generate_questions(client: genai.Client, course: CourseData) -> list[Question]:
    """
    Call Gemini to generate 20 MCQs from the course transcript.
    Uses structured JSON output for reliable parsing.
    """
    prompt = _build_generation_prompt(course)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text.strip()

    # Parse JSON response
    try:
        questions_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"  Failed to parse Gemini response as JSON: {e}")
        logger.debug(f"  Raw response (first 500 chars): {raw_text[:500]}")
        raise

    if not isinstance(questions_data, list):
        raise ValueError(f"Expected JSON array, got {type(questions_data).__name__}")

    if len(questions_data) != QUESTIONS_PER_COURSE:
        logger.warning(
            f"  Expected {QUESTIONS_PER_COURSE} questions, got {len(questions_data)}"
        )

    import re
    # Parse into Question objects
    questions = []
    for i, q in enumerate(questions_data):
        try:
            raw_text = q["question_text"]
            # Clean unwanted prefixes like "According to the transcript, " or "Based on the video,"
            clean_text = re.sub(r"^(According to|Based on|As stated in|In)( the)? (transcript|video|lesson|text|course)s?,\s*", "", raw_text, flags=re.IGNORECASE)
            # Capitalize first letter if it was lowercased by the strip
            if clean_text and clean_text[0].islower():
                clean_text = clean_text[0].upper() + clean_text[1:]

            questions.append(Question(
                question_text=clean_text,
                simplified_text=q["simplified_text"],
                question_type=q.get("question_type", "multiple_choice"),
                correct_answer=q["correct_answer"],
                options=q["options"],
                difficulty_level=int(q["difficulty_level"]),
            ))
        except (KeyError, ValueError) as e:
            logger.warning(f"  Skipping malformed question {i}: {e}")

    return questions


# ---------------------------------------------------------------------------
# Phase 3: Validate via LLM-as-a-Judge
# ---------------------------------------------------------------------------

def _build_validation_prompt(questions: list[Question], course: CourseData) -> str:
    """Build the validation prompt for the LLM judge."""
    questions_json = json.dumps(
        [
            {
                "index": i,
                "question_text": q.question_text,
                "simplified_text": q.simplified_text,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "difficulty_level": q.difficulty_level,
            }
            for i, q in enumerate(questions)
        ],
        indent=2,
    )

    return f"""You are a senior QA reviewer for educational assessments on the Tamkeen platform.

COURSE: {course.title}

Review these {len(questions)} questions. For each, check:
1. Is the correct_answer letter (A/B/C/D) actually pointing to the correct option?
2. Are distractors plausible but clearly wrong?
3. Is the difficulty_level appropriate? (1=basic recall, 2=understanding, 3=application, 4=analysis, 5=evaluation)
4. Is simplified_text genuinely simpler and shorter than question_text?
5. Are there any duplicate or near-duplicate questions?

QUESTIONS:
{questions_json}

Return ONLY a valid JSON object with this exact structure:
{{
  "valid": true or false,
  "flagged_indices": [list of integer indices of problematic questions],
  "issues": [
    {{"index": 0, "problem": "description", "suggestion": "how to fix"}}
  ],
  "difficulty_distribution": {{"1": N, "2": N, "3": N, "4": N, "5": N}}
}}

If all questions pass review, return valid=true with empty flagged_indices and issues."""


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda rs: logger.warning(
        f"  Retrying validation (attempt {rs.attempt_number})..."
    ),
)
def validate_questions(
    client: genai.Client,
    questions: list[Question],
    course: CourseData,
) -> ValidationResult:
    """Call Gemini as a judge to validate the generated questions."""

    prompt = _build_validation_prompt(questions, course)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,  # Low temperature for consistent judgment
            response_mime_type="application/json",
        ),
    )

    raw_text = response.text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"  Validation response not valid JSON: {e}")
        raise

    return ValidationResult(
        valid=result.get("valid", False),
        flagged_indices=result.get("flagged_indices", []),
        issues=result.get("issues", []),
        difficulty_distribution=result.get("difficulty_distribution", {}),
    )


# ---------------------------------------------------------------------------
# Phase 4: Save Output
# ---------------------------------------------------------------------------

def load_assessment_mapping() -> dict:
    """Load the assessment_id mapping file."""
    if not MAPPING_FILE.exists():
        logger.error(
            f"Assessment mapping file not found: {MAPPING_FILE}\n"
            f"Create it with course_slug -> assessment_id (int) mappings."
        )
        sys.exit(1)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Remove comments
    mapping.pop("_comment", None)

    # Validate
    unmapped = [k for k, v in mapping.items() if v is None]
    if unmapped:
        logger.warning(
            f"Unmapped courses (assessment_id is null): {unmapped}\n"
            f"These courses will use course_slug as a placeholder."
        )

    return mapping


def save_output(
    questions: list[Question],
    course: CourseData,
    assessment_mapping: dict,
) -> None:
    """Save questions as per-course JSON and append to combined CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assessment_id = assessment_mapping.get(course.slug)

    # Per-course JSON
    output_data = {
        "course_slug": course.slug,
        "course_title": course.title,
        "assessment_id": assessment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": GEMINI_MODEL,
        "total_questions": len(questions),
        "questions": [
            {
                "assessment_id": assessment_id,
                "question_text": q.question_text,
                "simplified_text": q.simplified_text,
                "question_type": q.question_type,
                "correct_answer": q.correct_answer,
                "options": q.options,
                "difficulty_level": q.difficulty_level,
                "isDeleted": False,
            }
            for q in questions
        ],
    }

    json_path = OUTPUT_DIR / f"{course.slug}_questions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"  → Saved: {json_path.name}")

    # Append to combined CSV
    csv_path = OUTPUT_DIR / "all_questions.csv"
    rows = []
    for q in questions:
        rows.append({
            "course_slug": course.slug,
            "assessment_id": assessment_id,
            "question_text": q.question_text,
            "simplified_text": q.simplified_text,
            "question_type": q.question_type,
            "correct_answer": q.correct_answer,
            "options": json.dumps(q.options, ensure_ascii=False),
            "difficulty_level": q.difficulty_level,
        })

    df = pd.DataFrame(rows)

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main():
    print()
    print("━" * 50)
    print("  Tamkeen Bulk Question Generator")
    print("━" * 50)
    print(f"  Source : {CONTENT_ROOT}")
    print(f"  Model : {GEMINI_MODEL}")
    print(f"  Output: {OUTPUT_DIR}")
    print("━" * 50)
    print()

    # Initialize Gemini client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY not found. Create a .env file with:\n"
            "  GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Load assessment mapping
    assessment_mapping = load_assessment_mapping()

    # Clean previous combined CSV
    csv_path = OUTPUT_DIR / "all_questions.csv"
    if csv_path.exists():
        csv_path.unlink()

    # Scan courses
    courses = scan_courses()
    if not courses:
        logger.error("No courses found. Check CONTENT_ROOT path.")
        sys.exit(1)

    print(f"\nFound {len(courses)} courses to process.\n")

    # Process each course
    total_generated = 0
    results_summary = []

    for i, course in enumerate(courses, 1):
        print(f"[{i}/{len(courses)}] {course.title} ({course.slug})")
        print(f"  Lessons: {len(course.lessons)}, "
              f"Chapters: {len(course.chapters)}, "
              f"Transcript: {course.total_chars:,} chars")

        # Check if already generated
        json_path = OUTPUT_DIR / f"{course.slug}_questions.json"
        if json_path.exists():
            print(f"  ✓ Already generated (skipping API calls)")
            with open(json_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                questions = [
                    Question(
                        question_text=q["question_text"],
                        simplified_text=q["simplified_text"],
                        question_type=q["question_type"],
                        correct_answer=q["correct_answer"],
                        options=q["options"],
                        difficulty_level=q["difficulty_level"],
                    )
                    for q in saved_data["questions"]
                ]
            save_output(questions, course, assessment_mapping)
            results_summary.append((course.slug, len(questions), "OK"))
            total_generated += len(questions)
            continue

        # Phase 2: Generate
        try:
            questions = generate_questions(client, course)
            print(f"  ✓ Generated {len(questions)} questions")
        except Exception as e:
            logger.error(f"  ✗ Generation failed: {e}")
            results_summary.append((course.slug, 0, "FAILED"))
            continue

        # Check difficulty distribution
        dist = {}
        for q in questions:
            dist[q.difficulty_level] = dist.get(q.difficulty_level, 0) + 1
        dist_str = "/".join(str(dist.get(d, 0)) for d in range(1, 6))
        print(f"  Distribution: {dist_str}")

        # Phase 3: Validate
        try:
            validation = validate_questions(client, questions, course)

            if validation.valid:
                print(f"  ✓ Validation passed")
            else:
                print(f"  ⚠ Validation flagged {len(validation.flagged_indices)} questions")
                for issue in validation.issues[:3]:  # Show first 3
                    print(f"    - Q{issue.get('index', '?')}: {issue.get('problem', 'unknown')}")

                # Retry: regenerate the entire batch (simpler than partial regen)
                logger.info("  Regenerating full batch...")
                questions = generate_questions(client, course)
                print(f"  ✓ Regenerated {len(questions)} questions")

        except Exception as e:
            logger.warning(f"  ⚠ Validation skipped (error: {e}), saving anyway.")

        # Phase 4: Save
        save_output(questions, course, assessment_mapping)
        total_generated += len(questions)
        results_summary.append((course.slug, len(questions), "OK"))
        print()

        # Brief pause between courses to respect rate limits
        if i < len(courses):
            time.sleep(2)

    # Final summary
    print("━" * 50)
    print("  Summary")
    print("━" * 50)
    for slug, count, status in results_summary:
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {slug:<25} {count:>3} questions  [{status}]")
    print(f"\n  Total: {total_generated} questions across {len(courses)} courses")
    print(f"  Output: {OUTPUT_DIR / 'all_questions.csv'}")
    print("━" * 50)


if __name__ == "__main__":
    main()
