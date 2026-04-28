import './style.css'

const API_URL = "http://127.0.0.1:8000/rag"

document.querySelector('#app').innerHTML = `
  <div class="page">
    <div class="header">
      <div>
        <div class="title">LexQuery</div>
        <div class="subtitle">Grounded legal RAG with source reader, review loop, and retrieval modes</div>
      </div>
    </div>

    <div class="toolbar">
      <label class="field">
        <span>Retrieval</span>
        <select id="retrievalMode">
          <option value="vector">Vector</option>
          <option value="page_index">pageIndex</option>
        </select>
      </label>

      <label class="field checkbox-field">
        <input id="reviewToggle" type="checkbox" checked />
        <span>LLM review loop</span>
      </label>
    </div>

    <div id="chat" class="chat"></div>

    <div class="composer">
      <textarea id="input" placeholder="Ask something..." rows="2"></textarea>
      <button id="send">Send</button>
    </div>
  </div>
`

const chatEl = document.getElementById("chat")
const inputEl = document.getElementById("input")
const sendBtn = document.getElementById("send")
const retrievalModeEl = document.getElementById("retrievalMode")
const reviewToggleEl = document.getElementById("reviewToggle")

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
}

function formatText(text) {
  if (!text) return ""
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>")
}

function renderReview(review, finalQueryUsed) {
  if (!review || !Array.isArray(review.rounds)) return ""

  const rounds = review.rounds.map(round => `
    <div class="review-round">
      <div class="review-line">
        <strong>Round ${escapeHtml(round.round_number)}</strong>
        <span class="review-verdict ${escapeHtml(round.verdict)}">${escapeHtml(round.verdict)}</span>
      </div>
      <div class="review-small">query: ${escapeHtml(round.query_used || "")}</div>
      <div class="review-small">scores: relevance ${escapeHtml(round.relevance_score ?? "-")}, groundedness ${escapeHtml(round.groundedness_score ?? "-")}, completeness ${escapeHtml(round.completeness_score ?? "-")}</div>
      <div class="review-small">${escapeHtml(round.rationale || "")}</div>
      ${round.rewritten_query ? `<div class="review-small">rewrite: ${escapeHtml(round.rewritten_query)}</div>` : ""}
    </div>
  `).join("")

  return `
    <details class="review-card">
      <summary>Review: ${escapeHtml(review.final_verdict || "unknown")} • final query: ${escapeHtml(finalQueryUsed || "")}</summary>
      ${rounds || `<div class="review-small">Review was skipped.</div>`}
    </details>
  `
}

function addMessage(role, content, citations = [], meta = {}) {
  const wrapper = document.createElement("div")
  wrapper.className = `message ${role}`

  const bubble = document.createElement("div")
  bubble.className = "bubble"

  const roleLabel = document.createElement("div")
  roleLabel.className = "role"
  roleLabel.textContent = role === "user" ? "You" : "Assistant"

  const text = document.createElement("div")
  text.className = "text"
  text.innerHTML = formatText(content)

  bubble.appendChild(roleLabel)
  bubble.appendChild(text)

  if (role === "assistant" && meta.review) {
    const reviewWrap = document.createElement("div")
    reviewWrap.innerHTML = renderReview(meta.review, meta.finalQueryUsed)
    bubble.appendChild(reviewWrap)
  }

  if (role === "assistant" && Array.isArray(citations) && citations.length > 0) {
    const citeTitle = document.createElement("div")
    citeTitle.className = "cite-title"
    citeTitle.textContent = "Sources"
    bubble.appendChild(citeTitle)

    citations.forEach(c => {
      const card = document.createElement("a")
      card.className = "citation citation-link"
      card.href = c.viewer_url || c.document_url || "#"
      card.target = "_blank"
      card.rel = "noreferrer noopener"
      card.innerHTML = `
        <div class="cite-header">
          <span class="badge">[${escapeHtml(c.index)}]</span>
          <strong>${escapeHtml(c.source_file)}</strong>
        </div>
        <div class="cite-meta">
          pages ${escapeHtml(String(c.page_start))}-${escapeHtml(String(c.page_end))}
          ${typeof c.score === "number" ? ` • score ${c.score.toFixed(4)}` : ""}
        </div>
        <div class="cite-preview">${escapeHtml(c.text_preview || "")}</div>
        <div class="cite-open">Open document at cited page</div>
      `
      bubble.appendChild(card)
    })
  }

  wrapper.appendChild(bubble)
  chatEl.appendChild(wrapper)
  chatEl.scrollTop = chatEl.scrollHeight
}

async function sendMessage() {
  const query = inputEl.value.trim()
  if (!query) return

  addMessage("user", query)
  inputEl.value = ""
  addMessage("assistant", "Thinking...")

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k: 5,
        temperature: 0.2,
        max_context_chars: 12000,
        stream: false,
        retrieval_mode: retrievalModeEl.value,
        enable_review: reviewToggleEl.checked,
        max_review_rounds: 2
      })
    })

    if (!res.ok) {
      const errText = await res.text().catch(() => "")
      throw new Error(`API Error ${res.status}: ${errText}`)
    }

    const data = await res.json()
    chatEl.lastChild?.remove()

    addMessage("assistant", data.answer, data.citations || [], {
      review: data.review,
      finalQueryUsed: data.final_query_used
    })
  } catch (err) {
    chatEl.lastChild?.remove()
    addMessage("assistant", "Error contacting RAG server.")
    console.error(err)
  }
}

sendBtn.addEventListener("click", sendMessage)

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
})
