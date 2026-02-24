// // import './style.css'
// // import viteLogo from '/vite.svg'
// // import javascriptLogo from './javascript.svg'

// // const API_URL = "http://127.0.0.1:8000/rag"

// // document.querySelector('#app').innerHTML = `
// //   <div class="page">
// //     <div class="header">
// //       <div class="title">Local RAG Chat</div>
// //       <div class="subtitle">FastAPI + Qdrant + Ollama</div>
// //     </div>

// //     <div id="chat" class="chat"></div>

// //     <div class="composer">
// //       <textarea id="input" placeholder="Ask something..." rows="2"></textarea>
// //       <button id="send">Send</button>
// //     </div>
// //   </div>
// // `

// // const chatEl = document.getElementById("chat")
// // const inputEl = document.getElementById("input")
// // const sendBtn = document.getElementById("send")

// // function addMessage(role, content, citations = []) {
// //   const wrapper = document.createElement("div")
// //   wrapper.className = `message ${role}`

// //   const bubble = document.createElement("div")
// //   bubble.className = "bubble"

// //   const roleLabel = document.createElement("div")
// //   roleLabel.className = "role"
// //   roleLabel.textContent = role === "user" ? "You" : "Assistant"

// //   const text = document.createElement("div")
// //   text.className = "text"
// //   text.innerHTML = formatText(content)

// //   bubble.appendChild(roleLabel)
// //   bubble.appendChild(text)

// //   if (role === "assistant" && citations.length > 0) {
// //     const citeTitle = document.createElement("div")
// //     citeTitle.className = "cite-title"
// //     citeTitle.textContent = "Citations"

// //     bubble.appendChild(citeTitle)

// //     citations.forEach(c => {
// //       const card = document.createElement("div")
// //       card.className = "citation"

// //       card.innerHTML = `
// //         <div class="cite-header">
// //           <span class="badge">[${c.index}]</span>
// //           <strong>${c.source_file}</strong>
// //         </div>
// //         <div class="cite-meta">
// //           pages ${c.page_start}-${c.page_end}
// //           ${typeof c.score === "number" ? ` • score ${c.score.toFixed(4)}` : ""}
// //         </div>
// //         <div class="cite-preview">${c.text_preview}</div>
// //       `

// //       bubble.appendChild(card)
// //     })
// //   }

// //   wrapper.appendChild(bubble)
// //   chatEl.appendChild(wrapper)

// //   chatEl.scrollTop = chatEl.scrollHeight
// // }

// // function formatText(text) {
// //   if (!text) return ""

// //   // bold **text**
// //   text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")

// //   // new lines
// //   text = text.replace(/\n/g, "<br/>")

// //   return text
// // }

// // async function sendMessage() {
// //   const query = inputEl.value.trim()
// //   if (!query) return

// //   addMessage("user", query)
// //   inputEl.value = ""

// //   addMessage("assistant", "Thinking...")

// //   try {
// //     const res = await fetch(API_URL, {
// //       method: "POST",
// //       headers: { "Content-Type": "application/json" },
// //       body: JSON.stringify({
// //         query,
// //         top_k: 5,
// //         temperature: 0.2,
// //         max_context_chars: 12000,
// //         stream: false
// //       })
// //     })

// //     if (!res.ok) {
// //       throw new Error("API Error: " + res.status)
// //     }

// //     const data = await res.json()

// //     // remove "Thinking..."
// //     chatEl.lastChild.remove()

// //     addMessage(
// //       "assistant",
// //       data.answer,
// //       Array.isArray(data.citations) ? data.citations : []
// //     )

// //   } catch (err) {
// //     chatEl.lastChild.remove()
// //     addMessage("assistant", "Error contacting RAG server.")
// //     console.error(err)
// //   }
// // }

// // sendBtn.addEventListener("click", sendMessage)

// // inputEl.addEventListener("keydown", (e) => {
// //   if (e.key === "Enter" && !e.shiftKey) {
// //     e.preventDefault()
// //     sendMessage()
// //   }
// // })
// import './style.css'
// import viteLogo from '/vite.svg'
// import javascriptLogo from './javascript.svg'

// const API_URL = "http://127.0.0.1:8000/rag"

// // How much memory to send each request (keep small for speed)
// const HISTORY_MAX_TURNS = 6 // 6 turns = 12 messages total if you store user+assistant separately
// const MAX_MSG_LEN = 4000    // must match backend ChatTurn max_length

// // Store conversation memory (what your FastAPI expects)
// const history = [] // items: { role: "user"|"assistant", content: "..." }

// document.querySelector('#app').innerHTML = `
//   <div class="page">
//     <div class="header">
//       <div class="title">Local RAG Chat</div>
//       <div class="subtitle">FastAPI + Qdrant + Ollama</div>
//     </div>

//     <div id="chat" class="chat"></div>

//     <div class="composer">
//       <textarea id="input" placeholder="Ask something..." rows="2"></textarea>
//       <button id="send">Send</button>
//     </div>
//   </div>
// `

// const chatEl = document.getElementById("chat")
// const inputEl = document.getElementById("input")
// const sendBtn = document.getElementById("send")

// function addMessage(role, content, citations = []) {
//   const wrapper = document.createElement("div")
//   wrapper.className = `message ${role}`

//   const bubble = document.createElement("div")
//   bubble.className = "bubble"

//   const roleLabel = document.createElement("div")
//   roleLabel.className = "role"
//   roleLabel.textContent = role === "user" ? "You" : "Assistant"

//   const text = document.createElement("div")
//   text.className = "text"
//   text.innerHTML = formatText(content)

//   bubble.appendChild(roleLabel)
//   bubble.appendChild(text)

//   if (role === "assistant" && citations.length > 0) {
//     const citeTitle = document.createElement("div")
//     citeTitle.className = "cite-title"
//     citeTitle.textContent = "Citations"
//     bubble.appendChild(citeTitle)

//     citations.forEach(c => {
//       const card = document.createElement("div")
//       card.className = "citation"

//       card.innerHTML = `
//         <div class="cite-header">
//           <span class="badge">[${c.index}]</span>
//           <strong>${escapeHtml(c.source_file)}</strong>
//         </div>
//         <div class="cite-meta">
//           pages ${escapeHtml(String(c.page_start))}-${escapeHtml(String(c.page_end))}
//           ${typeof c.score === "number" ? ` • score ${c.score.toFixed(4)}` : ""}
//         </div>
//         <div class="cite-preview">${escapeHtml(c.text_preview || "")}</div>
//       `
//       bubble.appendChild(card)
//     })
//   }

//   wrapper.appendChild(bubble)
//   chatEl.appendChild(wrapper)
//   chatEl.scrollTop = chatEl.scrollHeight
// }

// function formatText(text) {
//   if (!text) return ""
//   // escape first to avoid XSS, then add formatting
//   text = escapeHtml(text)
//   text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
//   text = text.replace(/\n/g, "<br/>")
//   return text
// }

// function escapeHtml(str) {
//   return String(str)
//     .replaceAll("&", "&amp;")
//     .replaceAll("<", "&lt;")
//     .replaceAll(">", "&gt;")
//     .replaceAll('"', "&quot;")
//     .replaceAll("'", "&#039;")
// }

// // Keep only last N turns (role+content entries)
// function getHistoryToSend() {
//   // backend wants list of ChatTurn: {role, content}
//   // keep last HISTORY_MAX_TURNS*2 messages (user+assistant pairs)
//   const maxMsgs = HISTORY_MAX_TURNS * 2
//   return history.slice(-maxMsgs)
// }

// function pushToHistory(role, content) {
//   const safe = (content || "").slice(0, MAX_MSG_LEN)
//   history.push({ role, content: safe })
// }

// async function sendMessage() {
//   const query = inputEl.value.trim()
//   if (!query) return

//   // UI
//   addMessage("user", query)
//   inputEl.value = ""

//   // Memory (store user turn)
//   pushToHistory("user", query)

//   // UI placeholder
//   addMessage("assistant", "Thinking...")

//   try {
//     const res = await fetch(API_URL, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         query,
//         top_k: 5,
//         temperature: 0.2,
//         max_context_chars: 12000,
//         stream: false,
//         history: getHistoryToSend() // ✅ THIS is the main change
//       })
//     })

//     if (!res.ok) {
//       const text = await res.text().catch(() => "")
//       throw new Error("API Error: " + res.status + " " + text)
//     }

//     const data = await res.json()

//     // remove "Thinking..."
//     chatEl.lastChild.remove()

//     addMessage(
//       "assistant",
//       data.answer,
//       Array.isArray(data.citations) ? data.citations : []
//     )

//     // Memory (store assistant turn)
//     pushToHistory("assistant", data.answer || "")

//   } catch (err) {
//     chatEl.lastChild.remove()
//     addMessage("assistant", "Error contacting RAG server.")
//     console.error(err)
//   }
// }

// sendBtn.addEventListener("click", sendMessage)

// inputEl.addEventListener("keydown", (e) => {
//   if (e.key === "Enter" && !e.shiftKey) {
//     e.preventDefault()
//     sendMessage()
//   }
// })


import './style.css'

const API_URL = "http://127.0.0.1:8000/rag"

document.querySelector('#app').innerHTML = `
  <div class="page">
    <div class="header">
      <div class="title">LexQuery</div>
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
  text = escapeHtml(text)
  text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
  text = text.replace(/\n/g, "<br/>")
  return text
}

function addMessage(role, content, citations = []) {
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

  if (role === "assistant" && Array.isArray(citations) && citations.length > 0) {
    const citeTitle = document.createElement("div")
    citeTitle.className = "cite-title"
    citeTitle.textContent = "Citations"
    bubble.appendChild(citeTitle)

    citations.forEach(c => {
      const card = document.createElement("div")
      card.className = "citation"

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
        stream: false
      })
    })

    if (!res.ok) {
      const errText = await res.text().catch(() => "")
      throw new Error(`API Error ${res.status}: ${errText}`)
    }

    const data = await res.json()

    // remove "Thinking..."
    chatEl.lastChild.remove()

    addMessage("assistant", data.answer, data.citations || [])
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