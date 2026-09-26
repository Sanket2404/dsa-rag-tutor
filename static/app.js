/**
 * DSA RAG Tutor - Frontend Controller
 * Handles Chat interactions, Document Ingestion, Speech-to-Text & Collapsible Sources
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const queryInput = document.getElementById("queryInput");
  const btnVoice = document.getElementById("btnVoice");
  const voiceWaves = document.getElementById("voiceWaves");
  const messagesContainer = document.getElementById("messagesContainer");
  const welcomeBox = document.getElementById("welcomeBox");
  
  const docList = document.getElementById("docList");
  const totalChunksBadge = document.getElementById("totalChunksBadge");
  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const uploadProgress = document.getElementById("uploadProgress");
  const progressBarFill = document.getElementById("progressBarFill");

  const topKSlider = document.getElementById("topKSlider");
  const topKVal = document.getElementById("topKVal");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdVal = document.getElementById("thresholdVal");
  const tempSlider = document.getElementById("tempSlider");
  const tempVal = document.getElementById("tempVal");

  const guardrailLabel = document.getElementById("guardrailLabel");
  const temperatureLabel = document.getElementById("temperatureLabel");
  const btnOpenHelpModal = document.getElementById("btnOpenHelpModal");
  const helpModal = document.getElementById("helpModal");
  const btnCloseHelpModal = document.getElementById("btnCloseHelpModal");

  const promptModal = document.getElementById("promptModal");
  const btnClosePromptModal = document.getElementById("btnClosePromptModal");
  const rawPromptContent = document.getElementById("rawPromptContent");

  const btnClearChat = document.getElementById("btnClearChat");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  const toast = document.getElementById("toast");

  let isSubmitting = false;

  // =========================================================================
  // Initialize App
  // =========================================================================
  loadStats();

  // Auto-resize textarea
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
  });

  // Slider adjustments
  topKSlider.addEventListener("input", (e) => {
    topKVal.textContent = e.target.value;
  });

  thresholdSlider.addEventListener("input", (e) => {
    thresholdVal.textContent = `${e.target.value}%`;
  });

  if (tempSlider && tempVal) {
    tempSlider.addEventListener("input", (e) => {
      tempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Mobile sidebar drawer & backdrop controls
  const sidebarBackdrop = document.getElementById("sidebarBackdrop");
  const sidebarCloseBtn = document.getElementById("sidebarCloseBtn");

  const closeSidebarMobile = () => {
    sidebar.classList.remove("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.remove("active");
  };

  const openSidebarMobile = () => {
    sidebar.classList.add("open");
    if (sidebarBackdrop) sidebarBackdrop.classList.add("active");
  };

  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      if (sidebar.classList.contains("open")) {
        closeSidebarMobile();
      } else {
        openSidebarMobile();
      }
    });
  }

  if (sidebarCloseBtn) {
    sidebarCloseBtn.addEventListener("click", closeSidebarMobile);
  }

  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", closeSidebarMobile);
  }

  // Clear chat
  btnClearChat.addEventListener("click", () => {
    messagesContainer.innerHTML = "";
    if (welcomeBox) {
      messagesContainer.appendChild(welcomeBox);
    }
    showToast("Chat reset.");
  });

  // Quick Prompt Chips
  document.querySelectorAll(".prompt-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query");
      queryInput.value = query;
      closeSidebarMobile();
      submitQuery(query);
    });
  });

  // =========================================================================
  // Chat Submission Handler
  // =========================================================================
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || isSubmitting) return;
    closeSidebarMobile();
    submitQuery(query);
  });

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  async function submitQuery(query) {
    isSubmitting = true;
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Hide welcome box if visible
    if (welcomeBox && welcomeBox.parentNode === messagesContainer) {
      welcomeBox.remove();
    }

    // 1. Append User Message
    appendUserMessage(query);

    // 2. Append AI Typing Indicator
    const typingIndicator = appendTypingIndicator();
    scrollToBottom();

    // 3. Request RAG Pipeline
    const topK = parseInt(topKSlider.value, 10);
    const threshold = parseFloat(thresholdSlider.value) / 100.0;
    const temperature = tempSlider ? parseFloat(tempSlider.value) : 0.10;

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          top_k: topK,
          relevance_threshold: threshold,
          temperature: temperature
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      typingIndicator.remove();

      // 4. Render AI Grounded Response
      appendAIMessage(data);
    } catch (err) {
      typingIndicator.remove();
      appendErrorMessage(`Failed to fetch answer: ${err.message}`);
    } finally {
      isSubmitting = false;
      scrollToBottom();
    }
  }

  // =========================================================================
  // Message Rendering
  // =========================================================================
  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="msg-avatar">You</div>
      <div class="msg-body">
        <div class="msg-content">${escapeHtml(text)}</div>
      </div>
    `;
    messagesContainer.appendChild(row);
  }

  function appendTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row ai";
    row.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-body">
        <div class="msg-content">
          <div class="typing-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
    return row;
  }

  function appendAIMessage(data) {
    const msgId = "msg-" + Date.now() + "-" + Math.random().toString(36).substring(2, 6);
    const row = document.createElement("div");
    row.className = `message-row ai ${!data.is_confident ? 'guardrail-hit' : ''}`;
    row.id = msgId;

    // Convert answer with rich Markdown, tables, code blocks
    let formattedAnswer = formatAnswerText(data.answer);

    // Guardrail Alert Banner if question was blocked
    let guardrailBanner = "";
    if (!data.is_confident && data.guardrail_status) {
      const topPct = (data.guardrail_status.top_score * 100).toFixed(1);
      const reqPct = Math.round(data.guardrail_status.threshold_applied * 100);
      guardrailBanner = `
        <div class="guardrail-alert">
          <span class="guardrail-badge">🛡️ Guardrail Blocked</span>
          <span>Highest match was <strong>${topPct}%</strong> (below <strong>${reqPct}%</strong> cutoff). Prevented hallucination.</span>
        </div>
      `;
    }

    // Sleek, Collapsible Accordion with Caret Button (Hidden by default!)
    let sourcesAccordionHtml = "";
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
      const itemsHtml = data.retrieved_chunks.map((c, i) => `
        <div class="source-item" data-index="${i}">
          <div class="source-item-top">
            <span class="source-doc-name">📄 [${i+1}] ${escapeHtml(c.source)}</span>
            <span class="score-badge">${c.relevance_percent}% Match</span>
          </div>
          <div class="source-item-preview">"${escapeHtml(c.content.replace(/\n/g, ' ').substring(0, 160))}..."</div>
        </div>
      `).join("");

      sourcesAccordionHtml = `
        <div class="sources-accordion">
          <button type="button" class="sources-caret-btn" aria-expanded="false" title="Click caret to view retrieved sources and match percentages">
            <span class="caret-icon-wrapper">
              <svg class="caret-svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </span>
            <span class="sources-title-text">Retrieved Sources & Excerpts</span>
            <span class="sources-count-badge">${data.retrieved_chunks.length}</span>
            <span class="sources-toggle-status">Inspect</span>
          </button>
          <div class="sources-drawer" style="display: none;">
            <div class="sources-chips">${itemsHtml}</div>
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-body">
        <div class="msg-content">
          ${guardrailBanner}
          <div class="answer-prose">${formattedAnswer}</div>
          ${sourcesAccordionHtml}
        </div>
        <div class="msg-actions">
          <span class="latency-tag">⚡ ${data.latency_ms}ms</span>
          <span class="temp-tag">🌡️ Temp: ${(data.temperature !== undefined ? data.temperature : 0.1).toFixed(2)}</span>
          <span class="engine-pill-tag">🤖 ${escapeHtml(data.generator_used || "Local Grounded Synthesizer")}</span>
          
          <button type="button" class="btn-copy-answer" title="Copy full response to clipboard">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span class="copy-text">Copy</span>
          </button>

          ${data.augmented_prompt ? `
            <button class="btn-inspect-prompt" title="View prompt passed to LLM">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>
              </svg>
              Inspect RAG Prompt
            </button>
          ` : ''}
        </div>
      </div>
    `;

    // 1. Caret Accordion Toggle Handler
    const caretBtn = row.querySelector(".sources-caret-btn");
    const drawer = row.querySelector(".sources-drawer");
    const toggleStatus = row.querySelector(".sources-toggle-status");
    if (caretBtn && drawer) {
      caretBtn.addEventListener("click", () => {
        const isExpanded = caretBtn.classList.contains("expanded");
        if (isExpanded) {
          caretBtn.classList.remove("expanded");
          caretBtn.setAttribute("aria-expanded", "false");
          drawer.style.display = "none";
          if (toggleStatus) toggleStatus.textContent = "Inspect";
        } else {
          caretBtn.classList.add("expanded");
          caretBtn.setAttribute("aria-expanded", "true");
          drawer.style.display = "block";
          if (toggleStatus) toggleStatus.textContent = "Hide";
        }
      });
    }

    // 2. Copy Answer Handler
    const copyBtn = row.querySelector(".btn-copy-answer");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(data.answer).then(() => {
          const textSpan = copyBtn.querySelector(".copy-text");
          if (textSpan) textSpan.textContent = "Copied!";
          copyBtn.classList.add("copied");
          setTimeout(() => {
            if (textSpan) textSpan.textContent = "Copy";
            copyBtn.classList.remove("copied");
          }, 2000);
        });
      });
    }

    // 3. Prompt Inspector Button
    const inspectBtn = row.querySelector(".btn-inspect-prompt");
    if (inspectBtn) {
      inspectBtn.addEventListener("click", () => {
        openPromptModal(data.augmented_prompt);
      });
    }

    // 4. Code Block Copy Buttons
    row.querySelectorAll(".btn-code-copy").forEach((codeCopyBtn) => {
      codeCopyBtn.addEventListener("click", (e) => {
        const codeEl = e.target.closest(".code-block-wrapper").querySelector("code");
        if (codeEl) {
          navigator.clipboard.writeText(codeEl.textContent).then(() => {
            codeCopyBtn.textContent = "Copied!";
            setTimeout(() => { codeCopyBtn.textContent = "Copy"; }, 1800);
          });
        }
      });
    });

    messagesContainer.appendChild(row);
  }

  function appendErrorMessage(errText) {
    const row = document.createElement("div");
    row.className = "message-row ai guardrail-hit";
    row.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-body">
        <div class="msg-content" style="color: #fca5a5;">
          ⚠️ ${escapeHtml(errText)}
        </div>
      </div>
    `;
    messagesContainer.appendChild(row);
  }

  function formatAnswerText(text) {
    if (!text) return "";

    // 1. If marked.js is available, use it for complete ChatGPT/Gemini Markdown rendering
    if (typeof marked !== "undefined") {
      try {
        marked.setOptions({
          breaks: true,
          gfm: true
        });
        let html = marked.parse(text);

        // Enhance <pre><code> blocks with ChatGPT-style top bar and Copy button
        html = html.replace(/<pre><code(?:\s+class="language-([^"]*)")?>([\s\S]*?)<\/code><\/pre>/gi, (match, lang, code) => {
          const displayLang = (lang || "code").trim();
          return `
            <div class="code-block-wrapper">
              <div class="code-block-header">
                <span class="code-lang-label">${escapeHtml(displayLang)}</span>
                <button type="button" class="btn-code-copy" aria-label="Copy code">Copy</button>
              </div>
              <pre><code class="language-${escapeHtml(displayLang)}">${code}</code></pre>
            </div>
          `;
        });

        // Wrap tables with horizontal scroll container for mobile
        html = html.replace(/<table>([\s\S]*?)<\/table>/gi, '<div class="table-scroll-wrapper"><table class="md-table">$1</table></div>');

        return html;
      } catch (err) {
        console.warn("Marked parse error, using fallback:", err);
      }
    }

    // 2. Comprehensive Fallback Parser
    const codeBlocks = [];
    let processed = text.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const idx = codeBlocks.length;
      const displayLang = lang.trim() || "code";
      codeBlocks.push(
        `<div class="code-block-wrapper">
          <div class="code-block-header">
            <span class="code-lang-label">${escapeHtml(displayLang)}</span>
            <button type="button" class="btn-code-copy">Copy</button>
          </div>
          <pre><code class="language-${escapeHtml(displayLang)}">${escapeHtml(code.trim())}</code></pre>
        </div>`
      );
      return `__CODE_BLOCK_${idx}__`;
    });

    processed = processed.replace(/^#### (.*?)$/gm, '<h4 class="md-h4">$1</h4>');
    processed = processed.replace(/^### (.*?)$/gm, '<h3 class="md-h3">$1</h3>');
    processed = processed.replace(/^## (.*?)$/gm, '<h2 class="md-h2">$1</h2>');

    const tableRegex = /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)/g;
    processed = processed.replace(tableRegex, (match, headerRow, bodyRows) => {
      const ths = headerRow.split("|").map(h => h.trim()).filter(h => h).map(h => `<th>${escapeHtml(h)}</th>`).join("");
      const rows = bodyRows.trim().split("\n").map(row => {
        const tds = row.split("|").map(c => c.trim()).filter(c => c).map(c => `<td>${escapeHtml(c)}</td>`).join("");
        return `<tr>${tds}</tr>`;
      }).join("");
      return `<div class="table-scroll-wrapper"><table class="md-table"><thead><tr>${ths}</tr></thead><tbody>${rows}</tbody></table></div>`;
    });

    processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    processed = processed.replace(/\*([^\*]+?)\*/g, '<em>$1</em>');
    processed = processed.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

    const lines = processed.split("\n");
    let result = "";
    let inList = false;
    let inOrderedList = false;

    lines.forEach((line) => {
      line = line.trim();
      if (!line) return;

      if (line.startsWith("__CODE_BLOCK_") || line.startsWith("<h2") || line.startsWith("<h3") || line.startsWith("<h4") || line.startsWith("<div class=\"table-scroll")) {
        if (inList) { result += "</ul>"; inList = false; }
        if (inOrderedList) { result += "</ol>"; inOrderedList = false; }
        result += line;
      } else if (line.startsWith("•") || line.startsWith("- ") || line.startsWith("* ")) {
        if (inOrderedList) { result += "</ol>"; inOrderedList = false; }
        if (!inList) { result += "<ul class=\"md-ul\">"; inList = true; }
        result += `<li>${line.replace(/^[•\-*]\s*/, '')}</li>`;
      } else if (/^\d+\.\s/.test(line)) {
        if (inList) { result += "</ul>"; inList = false; }
        if (!inOrderedList) { result += "<ol class=\"md-ol\">"; inOrderedList = true; }
        result += `<li>${line.replace(/^\d+\.\s*/, '')}</li>`;
      } else {
        if (inList) { result += "</ul>"; inList = false; }
        if (inOrderedList) { result += "</ol>"; inOrderedList = false; }
        result += `<p class="md-p">${line}</p>`;
      }
    });

    if (inList) result += "</ul>";
    if (inOrderedList) result += "</ol>";

    codeBlocks.forEach((block, idx) => {
      result = result.replace(`__CODE_BLOCK_${idx}__`, block);
    });

    return result;
  }

  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  // =========================================================================
  // Document Upload & Knowledge Base
  // =========================================================================
  uploadZone.addEventListener("click", () => fileInput.click());

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      uploadFiles(Array.from(e.dataTransfer.files));
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadFiles(Array.from(e.target.files));
    }
  });

  async function uploadFiles(files) {
    const validFiles = files.filter(f => {
      const ext = f.name.split('.').pop().toLowerCase();
      return ["pdf", "txt", "md"].includes(ext);
    });

    if (validFiles.length === 0) {
      showToast("Only .pdf, .txt, and .md files are supported!");
      return;
    }

    uploadProgress.style.display = "block";
    const uploadProgressText = document.getElementById("uploadProgressText");

    let totalUploaded = 0;
    let totalChunksAdded = 0;

    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i];
      const percent = Math.round(((i + 1) / validFiles.length) * 100);
      progressBarFill.style.width = `${percent}%`;
      if (uploadProgressText) {
        uploadProgressText.textContent = `Indexing ${i+1}/${validFiles.length}: ${file.name}...`;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        if (res.ok) {
          const result = await res.json();
          totalChunksAdded += result.chunks_added;
          totalUploaded++;
        }
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
      }
    }

    progressBarFill.style.width = "100%";
    if (uploadProgressText) {
      uploadProgressText.textContent = `✓ Indexed ${totalUploaded} files (${totalChunksAdded} chunks).`;
    }
    showToast(`✓ Successfully indexed ${totalUploaded} lecture PDFs!`);

    setTimeout(() => {
      uploadProgress.style.display = "none";
      progressBarFill.style.width = "0%";
      if (uploadProgressText) uploadProgressText.textContent = "Indexing 0 of 0...";
    }, 2500);

    loadStats();
    fileInput.value = "";
  }

  async function loadStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();

      totalChunksBadge.textContent = `${data.total_chunks} chunks`;

      if (data.indexed_documents && data.indexed_documents.length > 0) {
        docList.innerHTML = data.indexed_documents.map(doc => {
          const isPdf = doc.toLowerCase().endsWith(".pdf");
          return `
            <div class="doc-item">
              <span class="doc-icon">${isPdf ? '📕' : '📄'}</span>
              <span class="doc-name" title="${escapeHtml(doc)}">${escapeHtml(doc)}</span>
              <button class="btn-doc-delete" data-filename="${escapeHtml(doc)}" title="Delete document & purge vectors">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
              </button>
            </div>
          `;
        }).join("");

        // Bind delete click handlers
        docList.querySelectorAll(".btn-doc-delete").forEach(btn => {
          btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const filename = btn.getAttribute("data-filename");
            if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis will remove the file from disk and purge all its vector chunks from ChromaDB.`)) {
              return;
            }
            try {
              btn.disabled = true;
              btn.style.opacity = "0.3";
              const delRes = await fetch(`/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE"
              });
              if (!delRes.ok) {
                const err = await delRes.json();
                throw new Error(err.detail || "Delete failed");
              }
              const resData = await delRes.json();
              showToast(`✓ Deleted "${filename}" (${resData.chunks_deleted} chunks purged)`);
              loadStats();
            } catch (err) {
              showToast(`Delete failed: ${err.message}`);
              btn.disabled = false;
              btn.style.opacity = "1";
            }
          });
        });
      } else {
        docList.innerHTML = `<div class="loading-state">No documents indexed yet.</div>`;
      }
    } catch (err) {
      console.error("Failed to load stats:", err);
    }
  }

  const btnSyncFolder = document.getElementById("btnSyncFolder");
  if (btnSyncFolder) {
    btnSyncFolder.addEventListener("click", async () => {
      btnSyncFolder.disabled = true;
      btnSyncFolder.style.opacity = "0.5";
      showToast("Scanning sample_docs for new files...");
      try {
        const res = await fetch("/api/sync", { method: "POST" });
        if (!res.ok) throw new Error("Sync failed");
        const data = await res.json();
        showToast(data.message);
        loadStats();
      } catch (err) {
        showToast(`Sync error: ${err.message}`);
      } finally {
        btnSyncFolder.disabled = false;
        btnSyncFolder.style.opacity = "1";
      }
    });
  }

  // =========================================================================
  // Voice Input / Speech Recognition Controller (Live Waves & Auto-Pause)
  // =========================================================================
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;
  let silenceTimer = null;
  let baseQueryText = "";

  if (SpeechRecognition && btnVoice) {
    try {
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = navigator.language || "en-US";

      recognition.onstart = () => {
        isListening = true;
        btnVoice.classList.add("listening");
        if (voiceWaves) voiceWaves.style.display = "flex";
        baseQueryText = queryInput.value.trim() ? queryInput.value.trim() + " " : "";
        queryInput.setAttribute("placeholder", "🎙️ Listening to your voice... Speak now (pause when finished)");
        showToast("🎙️ Listening... Speak your question");
      };

      recognition.onresult = (event) => {
        // Clear silence detection timer on each new speech chunk
        clearTimeout(silenceTimer);

        let interimTranscript = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const transcriptChunk = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcriptChunk;
          } else {
            interimTranscript += transcriptChunk;
          }
        }

        const currentSpoken = (finalTranscript + interimTranscript).trim();
        if (currentSpoken) {
          queryInput.value = baseQueryText + currentSpoken;
          // Auto-resize input textarea to fit text
          queryInput.style.height = "auto";
          queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + "px";
        }

        // Automatic Pause Detection: when user pauses speaking for 1.3 seconds, finalize!
        silenceTimer = setTimeout(() => {
          if (isListening) {
            stopVoiceRecognition();
          }
        }, 1300);
      };

      recognition.onspeechend = () => {
        // Natural speech boundary detected by browser audio engine
        clearTimeout(silenceTimer);
        silenceTimer = setTimeout(() => {
          if (isListening) {
            stopVoiceRecognition();
          }
        }, 800);
      };

      recognition.onerror = (event) => {
        console.warn("[Voice Recognition Error]", event.error);
        if (event.error === "not-allowed") {
          showToast("Microphone permission denied. Please allow mic access in your browser.");
        } else if (event.error !== "no-speech") {
          showToast(`Voice input: ${event.error}`);
        }
        stopVoiceRecognition();
      };

      recognition.onend = () => {
        resetVoiceUI();
      };

      btnVoice.addEventListener("click", () => {
        if (isListening) {
          stopVoiceRecognition();
        } else {
          startVoiceRecognition();
        }
      });
    } catch (e) {
      console.warn("Could not initialize SpeechRecognition:", e);
    }
  } else if (btnVoice) {
    btnVoice.addEventListener("click", () => {
      showToast("Speech recognition is supported in Chrome, Safari, and Edge. Please open in a supported browser.");
    });
  }

  function startVoiceRecognition() {
    if (!recognition) return;
    try {
      recognition.start();
    } catch (err) {
      console.warn("Recognition start issue:", err);
    }
  }

  function stopVoiceRecognition() {
    clearTimeout(silenceTimer);
    if (!recognition || !isListening) return;
    try {
      recognition.stop();
    } catch (err) {
      // Ignore
    }
    resetVoiceUI();
  }

  function resetVoiceUI() {
    isListening = false;
    clearTimeout(silenceTimer);
    if (btnVoice) btnVoice.classList.remove("listening");
    if (voiceWaves) voiceWaves.style.display = "none";
    queryInput.setAttribute("placeholder", "Ask a question about your indexed documents...");
    
    if (queryInput.value.trim()) {
      queryInput.focus();
      showToast("🎙️ Voice captured! Press Enter or Send.");
    }
  }

  // Help Modal (Guardrail & Temperature Guide)
  const openHelpModal = () => { if (helpModal) helpModal.style.display = "flex"; };
  if (btnOpenHelpModal) btnOpenHelpModal.addEventListener("click", openHelpModal);
  if (guardrailLabel) guardrailLabel.addEventListener("click", openHelpModal);
  if (temperatureLabel) temperatureLabel.addEventListener("click", openHelpModal);
  if (btnCloseHelpModal) btnCloseHelpModal.addEventListener("click", () => { helpModal.style.display = "none"; });

  // Prompt Modal
  function openPromptModal(promptText) {
    rawPromptContent.textContent = promptText;
    promptModal.style.display = "flex";
  }
  btnClosePromptModal.addEventListener("click", () => { promptModal.style.display = "none"; });

  // Close modals on backdrop click
  window.addEventListener("click", (e) => {
    if (e.target === promptModal) promptModal.style.display = "none";
    if (e.target === helpModal) helpModal.style.display = "none";
  });

  // Toast Helper
  function showToast(msg) {
    toast.textContent = msg;
    toast.style.display = "block";
    setTimeout(() => {
      toast.style.display = "none";
    }, 2800);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>"']/g, (m) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    })[m]);
  }
});
