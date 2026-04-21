// Micro-Learning Curriculum Builder — Frontend Logic

class App {
    constructor() {
        this.currentView = 'pipeline';
        this.sessions = {};
        this.progress = {};
        this.currentSessionId = null;
        this.pipelineState = null;
        this.pipelineInterval = null;
        this.scrollListener = null;
        this._progressHandler = null;
        this.isStaticMode = false;
        this.currentBook = null;

        this.init();
    }

    async init() {
        // Detect static mode: no Flask backend available
        try {
            const ctrl = new AbortController();
            const t = setTimeout(() => ctrl.abort(), 1500);
            const test = await fetch('/api/state', { signal: ctrl.signal });
            clearTimeout(t);
            this.isStaticMode = !test.ok;
        } catch {
            this.isStaticMode = true;
        }

        await this.loadState();
        this.setupEventListeners();
        this.updateUI();

        if (this.currentView === 'pipeline') {
            this.startPipelinePolling();
        }

        this.initReadingProgress();
        this.setupSwipeGestures();
        this.initKeyboardNav();
    }

    _bookParam() {
        return this.currentBook ? `?book=${encodeURIComponent(this.currentBook)}` : '';
    }

    async loadState() {
        if (this.isStaticMode) {
            await this._loadStaticState();
            return;
        }
        try {
            const p = this._bookParam();
            const [stateRes, sessionsRes, progressRes] = await Promise.all([
                fetch(`/api/state${p}`),
                fetch(`/api/sessions${p}`),
                fetch(`/api/progress${p}`)
            ]);

            if (stateRes.ok) {
                this.pipelineState = await stateRes.json();
            }
            if (sessionsRes.ok) {
                const data = await sessionsRes.json();
                this.sessions = data.sessions || {};
            }
            if (progressRes.ok) {
                this.progress = await progressRes.json();
            }

            const isComplete = this.pipelineState && this.pipelineState.current_stage === 'Complete';
            const hasSessions = Object.keys(this.sessions).length > 0;

            if (isComplete && hasSessions) {
                this.currentView = 'feed';
            }
        } catch (err) {
            console.error('Error loading state:', err);
        }
    }

    async _loadStaticState() {
        try {
            const res = await fetch('./sessions.json');
            if (res.ok) {
                this.sessions = await res.json();
            }
        } catch (err) {
            console.error('Error loading static sessions:', err);
        }
        this.progress = this._localLoadProgress();
        this.pipelineState = { current_stage: 'Complete' };
        if (Object.keys(this.sessions).length > 0) {
            this.currentView = 'feed';
        }
    }

    _localProgressKey() {
        return `codex_progress${this.currentBook ? '_' + this.currentBook : ''}`;
    }

    _localLoadProgress() {
        try {
            return JSON.parse(localStorage.getItem(this._localProgressKey()) || '{}');
        } catch { return {}; }
    }

    _localSaveProgress() {
        localStorage.setItem(this._localProgressKey(), JSON.stringify(this.progress));
    }

    setupEventListeners() {
        document.getElementById('approve-btn')?.addEventListener('click', () => this.approveIndex());
        document.getElementById('submit-edits-btn')?.addEventListener('click', () => this.submitEdits());
        document.getElementById('switch-to-feed-btn')?.addEventListener('click', () => this.switchView('feed'));

        document.getElementById('back-to-list-btn')?.addEventListener('click', () => this.showSessionList());
        // mark-complete-btn is rendered dynamically inside #completion-action — handler attached in updateCompletionStatus
        document.getElementById('prev-session-btn')?.addEventListener('click', () => this.navigateSession('prev'));
        document.getElementById('next-session-btn')?.addEventListener('click', () => this.navigateSession('next'));

        document.getElementById('mobile-prev-btn')?.addEventListener('click', () => this.navigateSession('prev'));
        document.getElementById('mobile-home-btn')?.addEventListener('click', () => this.showSessionList());
        document.getElementById('mobile-next-btn')?.addEventListener('click', () => this.navigateSession('next'));
    }

    updateUI() {
        if (this.currentView === 'pipeline') {
            document.getElementById('pipeline-view').classList.remove('hidden');
            document.getElementById('feed-view').classList.add('hidden');
            this.updatePipelineView();
        } else {
            document.getElementById('pipeline-view').classList.add('hidden');
            document.getElementById('feed-view').classList.remove('hidden');
            this.updateFeedView();
        }
    }

    switchView(view) {
        this.currentView = view;
        this.updateUI();
        if (view === 'pipeline') {
            this.startPipelinePolling();
        } else {
            this.stopPipelinePolling();
        }
    }

    // ── Pipeline View ──────────────────────────────────────────────────────────

    // Ordered pipeline stages matching backend STAGES list
    get PIPELINE_STAGES() {
        return [
            { key: 'Ingest',       label: '1. PDF Ingestion' },
            { key: 'Pass1',        label: '2. Concept Inventory' },
            { key: 'Cluster',      label: '3. Theme Clustering' },
            { key: 'Pass2',        label: '4. Deep Extraction' },
            { key: 'GraphBuild',   label: '5. Dependency Graph' },
            { key: 'IndexGen',     label: '6. Session Planning' },
            { key: 'AwaitApproval',label: '7. Approval Gate' },
            { key: 'SessionGen',   label: '8. Session Generation' },
            { key: 'Complete',     label: '9. Complete' },
        ];
    }

    updatePipelineView() {
        this.updateStageStatus();
        this.updateIndexSection();
        this.updateGenerationProgress();
    }

    updateStageStatus() {
        if (!this.pipelineState) return;

        const currentStage = this.pipelineState.current_stage || 'Not started';
        const completedStages = new Set(Object.keys(this.pipelineState.outputs || {}));
        const isComplete = currentStage === 'Complete';

        this.PIPELINE_STAGES.forEach((stage, index) => {
            const el = document.querySelector(`.stage[data-stage="${stage.key}"]`);
            if (!el) return;

            el.classList.remove('completed', 'running');
            const indicator = el.querySelector('.stage-indicator');
            const statusText = el.querySelector('.stage-status-text');

            const isDone = completedStages.has(stage.key) || isComplete;
            const isRunning = !isDone && currentStage === stage.key;

            if (isDone) {
                el.classList.add('completed');
                indicator.textContent = '✓';
                statusText.textContent = 'Complete';
            } else if (isRunning) {
                el.classList.add('running');
                indicator.textContent = '…';
                statusText.textContent = 'Running';
            } else {
                indicator.textContent = '';
                statusText.textContent = 'Pending';
            }
        });
    }

    updateIndexSection() {
        if (!this.pipelineState) return;

        // Index is stored under 'IndexGen' key
        const indexOutput = this.pipelineState.outputs?.IndexGen;
        const indexSection = document.getElementById('index-section');
        const indexContent = document.getElementById('index-content');
        const indexApproval = document.getElementById('index-approval');

        if (!indexOutput || !indexOutput.index) {
            indexSection.classList.add('hidden');
            return;
        }

        indexSection.classList.remove('hidden');

        indexContent.innerHTML = indexOutput.index.map(session => `
            <div class="index-session">
                <h3>Session ${session.session_number}: ${this._esc(session.title)}</h3>
                <div class="meta">${session.estimated_minutes} min</div>
                <div class="concepts">
                    ${(session.concepts || []).map(c => `
                        <div class="concept">
                            <span class="concept-name">→ ${this._esc(c.name)}:</span>
                            ${this._esc(c.description)}
                        </div>
                    `).join('')}
                </div>
                ${session.revisit ? `
                    <div class="revisit">
                        <span class="concept-name">↺ ${this._esc(session.revisit.name)}:</span>
                        (${this._esc(session.revisit.reason)})
                    </div>
                ` : ''}
            </div>
        `).join('');

        if (indexOutput.approved) {
            indexApproval.classList.add('hidden');
        } else {
            indexApproval.classList.remove('hidden');
        }
    }

    updateGenerationProgress() {
        if (!this.pipelineState) return;

        const currentStage = this.pipelineState.current_stage;
        const sessionResults = this.pipelineState.session_results || {};
        const sessionsGenerated = Object.keys(sessionResults).length;
        // Total sessions from IndexGen metadata
        const totalSessions = this.pipelineState.outputs?.IndexGen?.metadata?.total_sessions || 0;

        const generationProgress = document.getElementById('generation-progress');
        const progressFill = document.getElementById('progress-fill');
        const currentSessionEl = document.getElementById('current-session');
        const sessionCountEl = document.getElementById('session-count');
        const pipelineActions = document.getElementById('pipeline-actions');

        if (currentStage === 'SessionGen' && totalSessions > 0) {
            generationProgress.classList.remove('hidden');

            const pct = totalSessions > 0 ? (sessionsGenerated / totalSessions) * 100 : 0;
            progressFill.style.width = `${pct}%`;

            const lastKey = Object.keys(sessionResults).pop();
            if (lastKey) {
                const sd = sessionResults[lastKey];
                currentSessionEl.textContent = `Session ${sd.session_number}: ${sd.title || ''}`;
            }

            sessionCountEl.textContent = `${sessionsGenerated} / ${totalSessions} sessions`;
        } else {
            generationProgress.classList.add('hidden');
        }

        if (currentStage === 'Complete' && Object.keys(this.sessions).length > 0) {
            pipelineActions.classList.remove('hidden');
        } else {
            pipelineActions.classList.add('hidden');
        }
    }

    async approveIndex() {
        try {
            const res = await fetch('/api/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'approve' })
            });
            if (res.ok) {
                await this.loadState();
                this.updatePipelineView();
            }
        } catch (err) {
            console.error('Error approving index:', err);
            alert('Failed to approve index');
        }
    }

    async submitEdits() {
        const instructions = document.getElementById('edit-instructions').value.trim();
        if (!instructions) {
            alert('Please enter edit instructions');
            return;
        }
        try {
            const res = await fetch('/api/index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'edit', instructions })
            });
            if (res.ok) {
                document.getElementById('edit-instructions').value = '';
                await this.loadState();
                this.updatePipelineView();
                alert('Edit instructions submitted. Revisions will be applied.');
            }
        } catch (err) {
            console.error('Error submitting edits:', err);
            alert('Failed to submit edits');
        }
    }

    startPipelinePolling() {
        if (this.pipelineInterval) return;
        this.pipelineInterval = setInterval(async () => {
            await this.loadState();
            this.updatePipelineView();
            if (this.pipelineState?.current_stage === 'Complete') {
                this.stopPipelinePolling();
            }
        }, 5000);
    }

    stopPipelinePolling() {
        if (this.pipelineInterval) {
            clearInterval(this.pipelineInterval);
            this.pipelineInterval = null;
        }
    }

    // ── Feed View ──────────────────────────────────────────────────────────────

    async updateFeedView() {
        if (!this.isStaticMode) {
            await this._renderBookSelector();
        }
        this.showSessionList();
    }

    async _renderBookSelector() {
        try {
            const res = await fetch('/api/books');
            if (!res.ok) return;
            const { books } = await res.json();
            if (books.length <= 1) return;

            let selector = document.getElementById('book-selector');
            if (!selector) {
                selector = document.createElement('select');
                selector.id = 'book-selector';
                selector.className = 'book-selector';
                const header = document.querySelector('.feed-header .feed-title-block');
                if (header) header.appendChild(selector);
            }
            selector.innerHTML = books.map(b =>
                `<option value="${this._esc(b.slug)}" ${b.slug === this.currentBook ? 'selected' : ''}>${this._esc(b.slug)} (${b.session_count || 0})</option>`
            ).join('');
            selector.onchange = async () => {
                this.currentBook = selector.value === '_legacy' ? null : selector.value;
                await this.loadState();
                this.updateUI();
            };
        } catch {}
    }

    showSessionList() {
        // Tear down any IntersectionObserver from the reading view
        this._teardownAutoCompletion();

        document.getElementById('session-list-view').classList.remove('hidden');
        document.getElementById('reading-view').classList.add('hidden');
        this.currentSessionId = null;
        this.renderSessionList();
        this._updateMobileNav('list');
        this.updateReadingProgress(); // resets the progress bar
    }

    renderSessionList() {
        const sessionList = document.getElementById('session-list');
        const sessionIds = Object.keys(this.sessions).sort();

        if (sessionIds.length === 0) {
            sessionList.innerHTML = '<p style="text-align:center;color:#64748b;padding:2rem">No sessions available yet. Complete the pipeline first.</p>';
            return;
        }

        // Update feed-count header
        const feedCount = document.getElementById('feed-session-count');
        if (feedCount) {
            const completedCount = Object.values(this.progress).filter(p => p && p.completed).length;
            feedCount.textContent = `${completedCount} / ${sessionIds.length} complete`;
        }

        sessionList.innerHTML = sessionIds.map(id => {
            const session = this.sessions[id];
            const prog = this.progress[id] || {};

            // Prefer stored estimated_minutes; fall back to a rough word-count estimate
            const estMinutes = session.estimated_minutes
                ? `${session.estimated_minutes} min`
                : null;

            // Concept count from metadata
            const conceptCount = Array.isArray(session.concepts)
                ? `${session.concepts.length} concept${session.concepts.length !== 1 ? 's' : ''}`
                : null;

            const metaParts = [conceptCount, estMinutes].filter(Boolean);
            const metaText = metaParts.join(' · ');

            let stateClass = '';
            let stateIndicator = '';
            if (prog.completed) {
                stateClass = 'complete';   // .session-card.complete
                stateIndicator = '<span class="status-icon completed-icon" aria-label="Complete">&#10003;</span>';
            } else if (prog.bookmark !== undefined && prog.bookmark !== null) {
                stateClass = 'in-progress'; // .session-card.in-progress
                stateIndicator = '<span class="status-icon progress-dot" aria-label="In progress">&#9679;</span>';
            }

            // Zero-padded session number
            const numDisplay = String(id).padStart(2, '0');

            return `
                <div class="session-card ${stateClass}" data-session-id="${id}" role="button" tabindex="0">
                    <div class="session-number">${this._esc(numDisplay)}</div>
                    <div class="session-info">
                        <div class="session-title">${this._esc(session.title || 'Untitled')}</div>
                        ${metaText ? `<div class="session-meta">${this._esc(metaText)}</div>` : ''}
                    </div>
                    ${stateIndicator}
                </div>
            `;
        }).join('');

        sessionList.querySelectorAll('.session-card').forEach(card => {
            const openFn = () => this.openSession(card.dataset.sessionId);
            card.addEventListener('click', openFn);
            card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') openFn(); });
        });
    }

    async openSession(sessionId) {
        this._teardownAutoCompletion();
        this.currentSessionId = sessionId;

        // In static mode sessions are fully loaded; in local mode fetch from API
        let session;
        if (this.isStaticMode) {
            session = this.sessions[sessionId];
        } else {
            try {
                const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}${this._bookParam()}`);
                if (!res.ok) {
                    console.error(`Failed to load session ${sessionId}:`, res.status);
                    return;
                }
                session = await res.json();
            } catch (err) {
                console.error('Error fetching session:', err);
                return;
            }
        }

        if (!session || !session.content) {
            console.error('Session content not available for', sessionId);
            return;
        }

        document.getElementById('session-list-view').classList.add('hidden');
        document.getElementById('reading-view').classList.remove('hidden');

        // Progress indicator: "3 / 12"
        const totalSessions = Object.keys(this.sessions).length;
        const sessionIds = Object.keys(this.sessions).sort();
        const pos = sessionIds.indexOf(sessionId) + 1;
        document.getElementById('current-session-display').textContent = `${pos} / ${totalSessions}`;

        this.renderSessionContent(session);

        // Restore bookmark scroll position
        const prog = this.progress[sessionId] || {};
        if (prog.bookmark !== undefined && prog.bookmark !== null) {
            // Small delay so layout is settled before scrolling
            requestAnimationFrame(() => {
                requestAnimationFrame(() => this.scrollToBookmark(prog.bookmark));
            });
        } else {
            window.scrollTo(0, 0);
        }

        this.updateCompletionStatus(sessionId);
        this.setupAutoCompletion(sessionId);
        this._applyHighlighting();
        this._updateMobileNav('reading');
    }

    renderSessionContent(session) {
        const container = document.getElementById('session-content');
        const content = session.content || '';

        // Build HTML: detect triple-backtick code fences first, then process prose
        let wordIndex = 0;
        const html = this._parseContent(content, wordIndex);

        // Prepend session title as blog-headline h1
        const titleHtml = session.title
            ? `<h1 class="session-headline">${this._esc(session.title)}</h1>`
            : '';

        // Summary card (optional field — may not exist yet)
        let summaryCardHtml = '';
        if (session.summary_card && session.summary_card.trim()) {
            const cardLines = session.summary_card
                .split('\n')
                .filter(line => line.trim())
                .map(line => `<p>${this._esc(line.trim())}</p>`)
                .join('');
            summaryCardHtml = `<div class="summary-card">${cardLines}</div>`;
        }

        // Reflection prompt — uses first concept name if available
        const firstConceptName = (Array.isArray(session.concepts) && session.concepts.length > 0)
            ? session.concepts[0].name
            : null;
        const reflectionHtml = firstConceptName
            ? `<div class="reflection-prompt">Before moving on — how would you explain ${this._esc(firstConceptName)} to a colleague who understands systems but not ML? Two sentences, out loud or on paper.</div>`
            : '';

        container.innerHTML = titleHtml + html.markup + summaryCardHtml + reflectionHtml;

        // Attach bookmark click handlers to every .word span
        container.querySelectorAll('.word').forEach(el => {
            el.addEventListener('click', () => this.setBookmark(parseInt(el.dataset.wordIndex, 10)));
        });
    }

    // Returns { markup: string } — parses triple-backtick code fences and prose
    _parseContent(content, startIndex) {
        let wordIndex = startIndex || 0;
        const parts = [];

        // Split on ```...``` fences (non-greedy, dot matches newline via workaround)
        const fenceRe = /```([^\n]*)\n([\s\S]*?)```/g;
        let lastIndex = 0;
        let match;

        while ((match = fenceRe.exec(content)) !== null) {
            // Prose before this fence
            const proseBefore = content.slice(lastIndex, match.index);
            if (proseBefore) {
                const result = this._renderProse(proseBefore, wordIndex);
                parts.push(result.html);
                wordIndex = result.nextIndex;
            }

            // Code fence: lang hint in match[1], code body in match[2]
            const codeBody = match[2];
            // Look for a plain-language explanation: a line immediately after closing ``` that
            // starts with a lowercase letter or "This", "The", "Note" — treated as prose explanation
            // We find it after the fence by peeking ahead in content
            const afterFence = content.slice(match.index + match[0].length);
            const explanationMatch = afterFence.match(/^\s*\n([A-Za-z][^\n]{10,})\n/);
            let explanation = null;
            if (explanationMatch) {
                explanation = explanationMatch[1].trim();
                // Advance fenceRe past the explanation line so it isn't re-processed as prose
                fenceRe.lastIndex += explanationMatch[0].length;
                lastIndex = match.index + match[0].length + explanationMatch[0].length;
            } else {
                lastIndex = match.index + match[0].length;
            }

            const langAttr = match[1] ? ` class="language-${this._esc(match[1].trim())}"` : '';
            parts.push(`<pre><code${langAttr}>${this._esc(codeBody.trimEnd())}</code></pre>`);
            if (explanation) {
                parts.push(`<div class="code-explanation">${this._esc(explanation)}</div>`);
            }
        }

        // Remaining prose after last fence
        const remaining = content.slice(lastIndex);
        if (remaining) {
            const result = this._renderProse(remaining, wordIndex);
            parts.push(result.html);
        }

        return { markup: parts.join('') };
    }

    // Renders a block of plain prose: splits on double-newlines into paragraphs,
    // wraps each word in a bookmarkable <span>.
    _renderProse(text, startIndex) {
        let wordIndex = startIndex;
        const paragraphs = text.split(/\n\n+/);

        const html = paragraphs.map(para => {
            const trimmed = para.trim();
            if (!trimmed) return '';

            // Wrap each word in a span for bookmarking
            const parts = trimmed.split(/(\s+)/);
            const wrapped = parts.map(part => {
                if (!part.trim()) return part; // preserve whitespace as-is
                const idx = wordIndex++;
                return `<span class="word" data-word-index="${idx}">${this._esc(part)}</span>`;
            }).join('');

            return `<p>${wrapped}</p>`;
        }).filter(Boolean).join('');

        return { html, nextIndex: wordIndex };
    }

    setBookmark(wordIndex) {
        if (!this.currentSessionId) return;

        // Remove previous bookmark highlight
        document.querySelectorAll('.word.bookmarked').forEach(el => el.classList.remove('bookmarked'));

        const el = document.querySelector(`.word[data-word-index="${wordIndex}"]`);
        if (el) el.classList.add('bookmarked');

        // Update local state immediately
        if (!this.progress[this.currentSessionId]) this.progress[this.currentSessionId] = {};
        this.progress[this.currentSessionId].bookmark = wordIndex;

        this._saveBookmarkToServer(this.currentSessionId, wordIndex);
    }

    async _saveBookmarkToServer(sessionId, wordIndex) {
        if (this.isStaticMode) {
            if (!this.progress[sessionId]) this.progress[sessionId] = {};
            this.progress[sessionId].bookmark = wordIndex;
            this._localSaveProgress();
            return;
        }
        try {
            await fetch(`/api/progress${this._bookParam()}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, bookmark_word_index: wordIndex })
            });
        } catch (err) {
            console.error('Error saving bookmark:', err);
        }
    }

    scrollToBookmark(wordIndex) {
        const el = document.querySelector(`.word[data-word-index="${wordIndex}"]`);
        if (el) {
            el.classList.add('bookmarked');
            // Position bookmarked word at ~30% from top of viewport
            const targetY = el.getBoundingClientRect().top + window.scrollY - window.innerHeight * 0.3;
            window.scrollTo({ top: Math.max(0, targetY), behavior: 'smooth' });
        }
    }

    setupAutoCompletion(sessionId) {
        this._teardownAutoCompletion();

        // Inject a sentinel div at the bottom of the reading content (before footer)
        const footer = document.querySelector('.reading-footer');
        let sentinel = document.getElementById('scroll-sentinel');
        if (!sentinel) {
            sentinel = document.createElement('div');
            sentinel.id = 'scroll-sentinel';
            sentinel.style.height = '1px';
            footer.parentNode.insertBefore(sentinel, footer);
        }

        let dwellTimer = null;

        this._completionObserver = new IntersectionObserver(entries => {
            const entry = entries[0];
            if (entry.isIntersecting) {
                // Sentinel is visible — start 3-second dwell timer
                dwellTimer = setTimeout(() => {
                    const prog = this.progress[sessionId] || {};
                    if (!prog.completed) {
                        this.markCurrentSessionComplete();
                    }
                }, 3000);
            } else {
                // Scrolled away — cancel pending dwell
                if (dwellTimer) {
                    clearTimeout(dwellTimer);
                    dwellTimer = null;
                }
            }
        }, { threshold: 0.1 });

        this._completionObserver.observe(sentinel);
    }

    _teardownAutoCompletion() {
        if (this._completionObserver) {
            this._completionObserver.disconnect();
            this._completionObserver = null;
        }
        // Remove scroll listener if one was left over from before
        if (this.scrollListener) {
            window.removeEventListener('scroll', this.scrollListener);
            this.scrollListener = null;
        }
    }

    updateCompletionStatus(sessionId) {
        const prog = this.progress[sessionId] || {};
        const completionAction = document.getElementById('completion-action');

        if (prog.completed) {
            completionAction.classList.add('hidden');
        } else {
            // Build ceremony block with concept chips (up to 3)
            const session = this.sessions[sessionId] || {};
            const concepts = Array.isArray(session.concepts) ? session.concepts : [];
            const chips = concepts
                .slice(0, 3)
                .map(c => this._esc(c.name || ''))
                .filter(Boolean)
                .join(' · ');

            const chipsHtml = chips
                ? `<p class="concepts-covered">You now hold: <span class="concept-chips">${chips}</span></p>`
                : '';

            completionAction.innerHTML = `
                <div class="completion-ceremony">
                    ${chipsHtml}
                    <button class="btn btn-primary mark-complete-btn">Mark complete →</button>
                </div>
            `;

            // Attach handler to the newly rendered button
            completionAction.querySelector('.mark-complete-btn')
                .addEventListener('click', () => this.markCurrentSessionComplete());

            completionAction.classList.remove('hidden');
        }

        const sessionIds = Object.keys(this.sessions).sort();
        const idx = sessionIds.indexOf(sessionId);
        document.getElementById('prev-session-btn').style.visibility = idx > 0 ? 'visible' : 'hidden';
        document.getElementById('next-session-btn').style.visibility = idx < sessionIds.length - 1 ? 'visible' : 'hidden';
    }

    async markCurrentSessionComplete() {
        if (!this.currentSessionId) return;
        const id = this.currentSessionId;

        // Prevent double-firing (e.g. button + IntersectionObserver)
        const prog = this.progress[id] || {};
        if (prog.completed) return;

        if (!this.progress[id]) this.progress[id] = {};
        this.progress[id].completed = true;
        delete this.progress[id].bookmark;

        if (this.isStaticMode) {
            this._localSaveProgress();
        } else {
            try {
                await fetch(`/api/progress${this._bookParam()}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: id,
                        completed: true,
                        bookmark_word_index: null
                    })
                });
            } catch (err) {
                console.error('Error saving completion:', err);
            }
        }

        this.updateCompletionStatus(id);
        document.querySelectorAll('.word.bookmarked').forEach(el => el.classList.remove('bookmarked'));
        this._teardownAutoCompletion();
        this._showToast('Session complete');
        this.renderSessionList();
    }

    _showToast(message) {
        let toast = document.getElementById('session-complete-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'session-complete-toast';
            toast.className = 'session-complete-toast';
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    }

    navigateSession(direction) {
        if (!this.currentSessionId) return;
        const ids = Object.keys(this.sessions).sort();
        const idx = ids.indexOf(this.currentSessionId);
        if (direction === 'prev' && idx > 0) this.openSession(ids[idx - 1]);
        else if (direction === 'next' && idx < ids.length - 1) this.openSession(ids[idx + 1]);
    }

    // ── Reading Progress Bar ──────────────────────────────────
    initReadingProgress() {
        this._progressHandler = () => this.updateReadingProgress();
        window.addEventListener('scroll', this._progressHandler, { passive: true });
    }

    updateReadingProgress() {
        const bar = document.getElementById('reading-progress-bar');
        if (!bar) return;
        const isReading = this.currentSessionId && !document.getElementById('reading-view')?.classList.contains('hidden');
        if (!isReading) {
            bar.style.width = '0%';
            bar.classList.remove('active');
            return;
        }
        bar.classList.add('active');
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const pct = docHeight > 0 ? Math.min(100, (scrollTop / docHeight) * 100) : 0;
        bar.style.width = `${pct}%`;
    }

    // ── Swipe Gestures (mobile) ───────────────────────────────
    setupSwipeGestures() {
        let startX = 0, startY = 0;
        document.addEventListener('touchstart', e => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        }, { passive: true });
        document.addEventListener('touchend', e => {
            if (!this.currentSessionId) return;
            if (document.getElementById('reading-view')?.classList.contains('hidden')) return;
            const dx = e.changedTouches[0].clientX - startX;
            const dy = e.changedTouches[0].clientY - startY;
            // Only count horizontal swipes (dx > dy to avoid scroll conflicts)
            if (Math.abs(dx) < 55 || Math.abs(dx) < Math.abs(dy) * 1.5) return;
            if (dx < 0) this.navigateSession('next');
            else this.navigateSession('prev');
        }, { passive: true });
    }

    // ── Keyboard Navigation ───────────────────────────────────
    initKeyboardNav() {
        document.addEventListener('keydown', e => {
            // Don't fire when typing in inputs/textareas
            if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
            if (!this.currentSessionId) return;
            if (document.getElementById('reading-view')?.classList.contains('hidden')) return;
            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                this.navigateSession('next');
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                this.navigateSession('prev');
            } else if (e.key === 'Escape') {
                this.showSessionList();
            }
        });
    }

    // ── Syntax Highlighting ───────────────────────────────────
    _applyHighlighting() {
        if (typeof hljs === 'undefined') return;
        document.querySelectorAll('#session-content pre code').forEach(block => {
            if (block.dataset.highlighted) return;
            hljs.highlightElement(block);
            const pre = block.closest('pre');
            if (!pre) return;
            // Extract language from class (highlight.js adds "language-xxx")
            const langMatch = block.className.match(/\blanguage-(\w+)\b/);
            if (langMatch) {
                pre.setAttribute('data-lang', langMatch[1]);
            } else if (block.result && block.result.language) {
                pre.setAttribute('data-lang', block.result.language);
            }
        });
    }

    // ── Mobile Bottom Nav visibility ──────────────────────────
    _updateMobileNav(mode) {
        const nav = document.getElementById('mobile-bottom-nav');
        if (!nav) return;
        if (mode === 'reading') {
            nav.classList.add('reading-mode');
            nav.setAttribute('aria-hidden', 'false');
        } else {
            nav.classList.remove('reading-mode');
            nav.setAttribute('aria-hidden', 'true');
        }
    }

    // ── Utilities ──────────────────────────────────────────────────────────────

    _esc(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
