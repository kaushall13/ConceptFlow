"""
Pipeline Orchestrator - Stage management with resume capability and progress display
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from state.manager import StateManager, get_state_manager
from api.cerebras import get_cerebras_client, DailyQuotaExhausted
from api.groq_client import get_groq_client
from api.ollama import get_ollama_client


class PipelineOrchestrator:
    """Manages the 9-stage curriculum generation pipeline."""

    STAGES = [
        "Ingest",
        "Pass1",
        "Cluster",
        "Pass2",
        "GraphBuild",
        "IndexGen",
        "AwaitApproval",
        "SessionGen",
        "Complete"
    ]

    def __init__(self, pdf_path: str, config: Dict[str, Any], auto_continue: bool = False, book_slug: str = None):
        self.pdf_path = pdf_path
        self.config = config
        self.auto_continue = auto_continue
        self.book_slug = book_slug
        self.state_manager = StateManager(book_slug=book_slug)

        self.cerebras_client = get_cerebras_client(config)
        self.groq_client = get_groq_client(config)

        self.ollama_client = get_ollama_client(
            host=config.get('ollama_host', 'http://localhost:11434'),
            model=config.get('ollama_model', 'llama3.2:3b')
        )

        if self.groq_client:
            print("  [Groq] Groq client initialized - will use for small tasks")
        else:
            print("  [Groq] No Groq API key configured - using Cerebras only")

    def _get_llm(self, prefer_groq: bool = False):
        """
        Return the best LLM for the task.
        prefer_groq=True for small tasks (titles, scoring) to save Cerebras quota.
        Falls back to Cerebras if Groq not available.
        """
        if prefer_groq and self.groq_client:
            return self.groq_client
        return self.cerebras_client

    def run(self):
        """Run the pipeline with resume capability."""
        current_stage = self.state_manager.get_current_stage()

        if current_stage == "Complete":
            print("Pipeline is already complete!")
            print("To start fresh, run: python main.py --reset")
            return

        if current_stage not in ("Not started", "Complete"):
            progress_info = self.state_manager.get_progress_info()
            print(f"\nPipeline is currently at stage: {current_stage}")
            print(f"Completed stages: {', '.join(progress_info['completed_stages'])}")
            print(f"Sessions generated: {progress_info['sessions_generated']}/{progress_info['total_sessions']}")

            if not self.auto_continue:
                try:
                    resume = input("\nResume from current stage? [y/n]: ").strip().lower()
                except EOFError:
                    print("Non-interactive environment - resuming...")
                    resume = 'y'

                if resume != 'y':
                    print("Starting fresh pipeline...")
                    self.state_manager.reset_all()
                    current_stage = "Not started"
            else:
                print(f"Auto-continuing from stage: {current_stage}")

        try:
            self._run_pipeline_stages(current_stage)
        except KeyboardInterrupt:
            print("\n\nPipeline interrupted. Progress has been saved.")
            print(f"Resume with: python main.py --pipeline {self.pdf_path}")
            sys.exit(0)
        except DailyQuotaExhausted:
            self._handle_quota_exhausted()

    def _handle_quota_exhausted(self):
        """Save progress and exit cleanly when all daily API quotas are exhausted."""
        progress_info = self.state_manager.get_progress_info()
        current_stage = progress_info["current_stage"]
        sessions_generated = progress_info["sessions_generated"]
        total_sessions = progress_info["total_sessions"]

        divider = "\u2501" * 40  # ━━━━…
        print(f"\n{divider}")
        print("Daily API quota exhausted for all keys.")
        if total_sessions:
            print(
                f"Progress saved at: {current_stage} "
                f"\u2014 session {sessions_generated} / {total_sessions}"
            )
        else:
            print(f"Progress saved at: {current_stage}")
        print(f"Resume tomorrow with: python main.py --pipeline {self.pdf_path}")
        print(divider)
        sys.exit(0)

    def _run_pipeline_stages(self, start_stage: str):
        stages_to_run = self._get_stages_to_run(start_stage)

        for stage in stages_to_run:
            self._run_stage(stage)

        print("\n" + "="*60)
        print("SUCCESS: PIPELINE COMPLETE")
        print("="*60)

    def _get_stages_to_run(self, current_stage: str) -> list:
        if current_stage == "Not started":
            return self.STAGES[:-1]  # All except "Complete" marker

        try:
            current_index = self.STAGES.index(current_stage)
            return self.STAGES[current_index + 1:]
        except ValueError:
            print(f"Warning: Unknown stage '{current_stage}', starting from beginning")
            return self.STAGES[:-1]

    def _run_stage(self, stage: str):
        print(f"\n{'='*60}")
        print(f"STAGE: {stage}")
        print('='*60)

        if stage == "Ingest":
            self._run_ingest()
        elif stage == "Pass1":
            self._run_pass1()
        elif stage == "Cluster":
            self._run_cluster()
        elif stage == "Pass2":
            self._run_pass2()
        elif stage == "GraphBuild":
            self._run_graphbuild()
        elif stage == "IndexGen":
            self._run_indexgen()
        elif stage == "AwaitApproval":
            self._run_approval()
        elif stage == "SessionGen":
            self._run_sessiongen()
        elif stage == "Complete":
            self._mark_complete()

    def _run_ingest(self):
        print("Extracting and cleaning PDF content...")
        from pipeline.ingest import extract_and_clean_pdf
        result = extract_and_clean_pdf(self.pdf_path)
        self.state_manager.save_stage("Ingest", result)
        print(f"SUCCESS: Extracted {len(result)} characters of clean text")

    def _run_pass1(self):
        print("Performing structural read to identify concepts...")
        from pipeline.pass1 import perform_pass1
        clean_text = self.state_manager.get_stage_output("Ingest")
        result = perform_pass1(clean_text, self.cerebras_client)
        self.state_manager.save_stage("Pass1", result)
        print(f"SUCCESS: Identified {len(result['concepts'])} concepts")

    def _run_cluster(self):
        print("Clustering concepts into themes...")
        from pipeline.cluster import cluster_concepts
        concepts = self.state_manager.get_stage_output("Pass1")['concepts']
        if not self.ollama_client:
            raise RuntimeError(
                "Ollama is required for the Cluster stage but is unreachable. "
                "Ensure Ollama is running at " + self.config.get('ollama_host', 'http://localhost:11434')
            )
        result = cluster_concepts(concepts, self.ollama_client)
        self.state_manager.save_stage("Cluster", result)
        print(f"SUCCESS: Created {len(result['clusters'])} thematic clusters")

    def _run_pass2(self):
        print("Performing deep extraction per cluster...")
        from pipeline.pass2 import perform_pass2
        clean_text = self.state_manager.get_stage_output("Ingest")
        all_concepts = self.state_manager.get_stage_output("Pass1")['concepts']
        clusters = self.state_manager.get_stage_output("Cluster")['clusters']
        # Pass2 uses Cerebras (needs large context)
        result = perform_pass2(clean_text, all_concepts, clusters, self.cerebras_client)
        self.state_manager.save_stage("Pass2", result)
        total = sum(len(c.get('concepts', [])) for c in result.values())
        print(f"SUCCESS: Deep extracted {total} concepts across {len(result)} clusters")

    def _run_graphbuild(self):
        print("Building concept dependency graph...")
        from pipeline.graph import build_concept_graph
        pass2_output = self.state_manager.get_stage_output("Pass2")
        if not self.ollama_client:
            raise RuntimeError(
                "Ollama is required for the GraphBuild stage but is unreachable. "
                "Ensure Ollama is running at " + self.config.get('ollama_host', 'http://localhost:11434')
            )
        result = build_concept_graph(pass2_output, self.ollama_client)
        self.state_manager.save_stage("GraphBuild", result)
        print(f"SUCCESS: Built graph with {len(result['concepts'])} concepts, {len(result['edges'])} edges")

    def _run_indexgen(self):
        print("Planning sessions and generating index...")
        from pipeline.planner import plan_sessions
        graph_output = self.state_manager.get_stage_output("GraphBuild")
        if not self.ollama_client:
            raise RuntimeError(
                "Ollama is required for the IndexGen stage but is unreachable. "
                "Ensure Ollama is running at " + self.config.get('ollama_host', 'http://localhost:11434')
            )
        result = plan_sessions(graph_output, self.ollama_client)
        self.state_manager.save_stage("IndexGen", result)
        print(f"SUCCESS: Generated {len(result['index'])} sessions")
        print("\n" + "="*60)
        print("CURRICULUM INDEX")
        print("="*60)
        self._print_index(result['index'])
        print("="*60)

    def _run_approval(self):
        print("\nReview the curriculum index above.")

        while True:
            try:
                import sys
                if not sys.stdin.isatty():
                    print("Non-interactive environment - auto-approving...")
                    instruction = 'approve'
                else:
                    instruction = input("\nEnter edit instructions (or 'approve' to continue): ").strip()
            except EOFError:
                print("Non-interactive environment - auto-approving...")
                instruction = 'approve'

            if not instruction:
                print("Please enter instructions or type 'approve'")
                continue

            if instruction.lower() == 'approve':
                indexgen_output = self.state_manager.get_stage_output("IndexGen")
                indexgen_output['approved'] = True
                self.state_manager.save_stage("IndexGen", indexgen_output)
                print("SUCCESS: Index approved")
                break

            print(f"Processing revision: {instruction}")
            from pipeline.planner import revise_index
            indexgen_output = self.state_manager.get_stage_output("IndexGen")
            revised = revise_index(indexgen_output['index'], instruction, self.cerebras_client)
            indexgen_output['index'] = revised
            self.state_manager.save_stage("IndexGen", indexgen_output)

            print("\n" + "="*60)
            print("REVISED CURRICULUM INDEX")
            print("="*60)
            self._print_index(revised)
            print("="*60)

    def _run_sessiongen(self):
        from pipeline.generator import generate_session_content
        from pipeline.evaluator import evaluate_session

        indexgen_output = self.state_manager.get_stage_output("IndexGen")
        graph_output = self.state_manager.get_stage_output("GraphBuild")

        sessions = indexgen_output['index']
        total_sessions = len(sessions)

        print(f"\nGenerating {total_sessions} sessions sequentially...")

        for i, session_plan in enumerate(sessions, 1):
            session_num = session_plan['session_number']
            title = session_plan['title']

            # Skip sessions already successfully generated (resume support)
            existing = self.state_manager.get_session_result(session_num)
            if existing and existing.get('content'):
                print(f"\n[{i}/{total_sessions}] Skipping session {session_num} (already generated)")
                continue

            print(f"\n[{i}/{total_sessions}] Generating session {session_num}: {title}")

            try:
                session_text, tension_excerpt = generate_session_content(
                    session_plan,
                    graph_output,
                    self.state_manager,
                    self.cerebras_client
                )

                print(f"  Evaluating session {session_num}...")
                evaluation_results, session_text = evaluate_session(
                    session_text,
                    session_plan,
                    self.state_manager,
                    self.ollama_client,
                    self.cerebras_client
                )

                session_result = {
                    'session_number': session_num,
                    'title': title,
                    'content': session_text,
                    'tension_excerpt': tension_excerpt,
                    'evaluation': evaluation_results,
                    'needs_review': any(not r['passed'] for r in evaluation_results.values())
                }

                self.state_manager.save_session_result(session_num, session_result)
                self._print_evaluation_summary(evaluation_results)
                print(f"  SUCCESS: Session {session_num} complete")

            except DailyQuotaExhausted:
                # Re-raise immediately — don't swallow quota exhaustion per-session
                raise

            except Exception as e:
                print(f"  ERROR: Session {session_num} failed: {e}")
                import traceback
                traceback.print_exc()

                failed_result = {
                    'session_number': session_num,
                    'title': title,
                    'content': None,
                    'evaluation': {},
                    'needs_review': True,
                    'error': str(e)
                }
                self.state_manager.save_session_result(session_num, failed_result)

                try:
                    choice = input("  Continue with remaining sessions? [y/n]: ").strip().lower()
                except EOFError:
                    print("  Non-interactive - continuing...")
                    choice = 'y'

                if choice != 'y':
                    print("Pipeline stopped by user.")
                    sys.exit(1)

        self._export_sessions()
        print(f"\nSUCCESS: All {total_sessions} sessions generated")

    def _mark_complete(self):
        self.state_manager.mark_complete()
        print("SUCCESS: Pipeline marked as complete")

    def _export_sessions(self):
        import json as _json
        output_dir = Path(f"data/{self.book_slug}/output") if self.book_slug else Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        session_results = self.state_manager.get_session_results()

        sessions_export = {}
        for session_id, result in session_results.items():
            if result.get('content'):
                sessions_export[session_id] = {
                    'title': result['title'],
                    'content': result['content'],
                    'needs_review': result.get('needs_review', False),
                    'evaluation': result.get('evaluation', {})
                }

        output_file = output_dir / "sessions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            _json.dump(sessions_export, f, indent=2, ensure_ascii=False)

        print(f"  Sessions exported to {output_file}")

    def _print_index(self, index: list):
        for session in index:
            print(f"\nSession {session['session_number']}: {session['title']} ({session.get('estimated_minutes', '?')} min)")
            for concept in session.get('concepts', []):
                print(f"  -> {concept['name']}: {concept['description']}")
            if session.get('revisit'):
                revisit = session['revisit']
                print(f"  -> REVISIT: {revisit['name']} ({revisit['reason']})")

    def _print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        parts = []
        for check_name, result in evaluation_results.items():
            if result['passed']:
                parts.append(f"[PASS] {check_name}")
            else:
                rc = result.get('retry_count', 0)
                parts.append(f"[FAIL] {check_name}(retry {rc})")
        print(f"  Eval: {' | '.join(parts)}")


def run_orchestrator(pdf_path: str, config: Dict[str, Any], auto_continue: bool = False, book_slug: str = None):
    orchestrator = PipelineOrchestrator(pdf_path, config, auto_continue=auto_continue, book_slug=book_slug)
    orchestrator.run()
