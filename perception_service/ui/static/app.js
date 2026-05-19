const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const submitState = document.getElementById("submitState");
const sendButton = document.getElementById("sendButton");
const providerNameEl = document.getElementById("providerName");
const modelNameEl = document.getElementById("modelName");
const healthStatusEl = document.getElementById("healthStatus");
const healthDetailEl = document.getElementById("healthDetail");
const sessionIdEl = document.getElementById("sessionId");
const runMetaEl = document.getElementById("runMeta");
const resetSessionBtn = document.getElementById("resetSession");
const commitMemoryEl = document.getElementById("commitMemory");
const persistMemoryEl = document.getElementById("persistMemory");
const messageTemplate = document.getElementById("messageTemplate");

const SESSION_KEY = "ai-agent-prototype-session-id";

function generateSessionId() {
  return `session-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function getSessionId() {
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  sessionIdEl.textContent = sessionId;
  return sessionId;
}

function setSubmitState(text, busy = false) {
  submitState.textContent = text;
  sendButton.disabled = busy;
}

function appendMessage(role, body, details = null) {
  const fragment = messageTemplate.content.cloneNode(true);
  const article = fragment.querySelector(".message");
  article.classList.add(role);
  fragment.querySelector(".message-role").textContent = role === "user" ? "You" : "Agent";
  fragment.querySelector(".message-time").textContent = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
  fragment.querySelector(".message-body").textContent = body;

  const detailsNode = fragment.querySelector(".message-details");
  if (details) {
    fragment.querySelector(".details-pre").textContent = JSON.stringify(details, null, 2);
  } else {
    detailsNode.remove();
  }

  messagesEl.appendChild(fragment);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function loadRuntimeInfo() {
  try {
    const [configRes, healthRes] = await Promise.all([
      fetch("/config"),
      fetch("/health"),
    ]);

    const config = await configRes.json();
    const health = await healthRes.json();

    providerNameEl.textContent = config.llm_provider;
    modelNameEl.textContent = config.llm_model;
    healthStatusEl.textContent = health.status;
    healthDetailEl.textContent = health.ollama_reachable
      ? "Local model runtime reachable"
      : "Using fallback or local model not reachable";
  } catch (error) {
    providerNameEl.textContent = "unknown";
    modelNameEl.textContent = "unavailable";
    healthStatusEl.textContent = "error";
    healthDetailEl.textContent = error.message;
  }
}

function buildTurnInput(message) {
  return {
    turn_id: `turn-${Date.now()}`,
    session_id: getSessionId(),
    user_id: "local-user",
    timestamp: new Date().toISOString(),
    message_text: message,
    recent_context: {
      recent_turn_summaries: [],
      active_topic: null,
      unresolved_questions: [],
      conversation_mode: "technical_collaboration",
      last_assistant_action: null,
    },
  };
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }

  appendMessage("user", text);
  messageInput.value = "";
  setSubmitState("Running full agent pipeline...", true);

  try {
    const response = await fetch("/agent/run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        turn_input: buildTurnInput(text),
        commit_memory: commitMemoryEl.checked,
        persist_committed_memory: persistMemoryEl.checked,
        use_session_state: true,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Agent run failed");
    }

    appendMessage("assistant", payload.generation.output.response_text, {
      evaluation: payload.evaluation,
      critic: payload.critic,
      working_memory: payload.working_memory.state,
      committed_memories: payload.memory_commit?.committed_memories || [],
    });

    runMetaEl.textContent =
      `Provider: ${payload.generation.output.metadata.provider_name}\n` +
      `Model: ${payload.generation.output.metadata.model_name}\n` +
      `Critic score: ${payload.critic.evaluation.score_summary}\n` +
      `Retrieved memories: ${payload.prompt_assembly.prompt.retrieved_memories.length}\n` +
      `Committed memories: ${payload.memory_commit?.evaluation.committed_count || 0}`;
  } catch (error) {
    appendMessage("assistant", `Request failed: ${error.message}`);
  } finally {
    setSubmitState("Ready", false);
  }
});

resetSessionBtn.addEventListener("click", async () => {
  const oldSessionId = getSessionId();
  try {
    await fetch(`/session/${oldSessionId}`, { method: "DELETE" });
  } catch (error) {
    console.warn("Failed to clear prior session", error);
  }
  const nextSessionId = generateSessionId();
  localStorage.setItem(SESSION_KEY, nextSessionId);
  sessionIdEl.textContent = nextSessionId;
  messagesEl.innerHTML = "";
  runMetaEl.textContent = "Session reset. Ready for a fresh run.";
});

getSessionId();
loadRuntimeInfo();
