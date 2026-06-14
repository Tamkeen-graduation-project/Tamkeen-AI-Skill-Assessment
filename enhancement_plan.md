# Enhancement Plan: Tamkeen AI Skill Assessment (Production-Ready)

This document outlines the proposed enhancements for the Tamkeen Adaptive Skill Assessment project. The goal is to move from a prototype to a robust, scalable, and intelligent production system, focusing on a global content pipeline and a scalable API.

## 1. Global Dynamic Content Pipeline (Scale-Ready)
Transition from static, manual files to a centralized, automated engine that generates and manages question banks for all assessments.

- **Task: Automated Bulk Ingestion**: Create `scripts/bulk_question_generator.py` to:
    - Scan all course transcripts in `data/transcripts/`.
    - Map course-specific learning objectives to difficulty levels (1-5).
    - Generate 15-20 questions per course with automated distractor validation.
- **Task: LLM-as-a-Judge**: Implement a validation step where a secondary LLM call verifies the technical accuracy and difficulty calibration of every generated question.
- **Task: Persistent Question Bank**: Migrate all questions into a SQL Database (PostgreSQL) indexed by `course_id` and `difficulty_level`.

## 2. Advanced Adaptive Engine (Core Logic)
Refine the adaptive logic in the API to be more sensitive to user performance and accessibility needs.

- **Task**: Upgrade the `next_difficulty` logic in the API to incorporate:
    - **Response Time weighting**: If a user answers correctly and significantly faster than the disability-adjusted average, jump +2 difficulty.
    - **Persistence**: Track streaks (correct/incorrect) to adjust the "aggression" of difficulty changes.
- **Task**: Implement a "Stability Score" that measures how much a user's performance fluctuates, providing more context for the final skill prediction.

## 3. Production API Infrastructure
Transition from a simple FastAPI script to a professional service architecture capable of handling multiple courses simultaneously.

- **Task: Database Integration**: Replace the global `sessions` and `QUESTIONS` variables with a persistent database (PostgreSQL/SQLAlchemy).
- **Task: Asynchronous Processing**: Implement background tasks for model prediction and result logging to ensure sub-second response times.
- **Task: State Management**: Use Redis or a similar store for active session state to support horizontal scaling of the API.

## 4. Accessibility & Support Services
Provide backend services that enable frontends to offer superior accessibility.

- **Task: Question Simplification API**: Create an endpoint `POST /simplify_question` that uses an LLM to rephrase complex questions into "Plain Language" for cognitive disability support.
- **Task: Metadata Enrichment**: Add accessibility metadata to each question (e.g., `alt_text` for images, `audio_description` hints).

## 5. Model Refinement & Feedback Loop
Continuously improve the skill prediction model using real-world data across all assessment categories.

- **Task: Data Collection**: Implement a logging service that captures every interaction for future retraining.
- **Task: Drift Detection**: Monitor model performance over time to ensure that the "Disability Multipliers" remain accurate as user demographics evolve.

---

### Suggested Next Step:
Would you like me to start by building the **Bulk Generation Script** (to process all transcripts at once) or should we first set up the **SQL Database** to store the generated questions?
