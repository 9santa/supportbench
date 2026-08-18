"use strict";

let currentWorld = null;
let currentRun = null;
let busy = false;

const MIXED_RAG_TASK =
    "Which DASH version is currently installed on dash-host-01? Search IBM support documentation for Web GUI 8.1 requirements and report only what the available evidence establishes.";
const WRITE_APPROVAL_TASK =
    "Check the current production Web GUI service state and open a support case if it is not fully operational.";

const elements = {
    errorPanel: document.querySelector("#error-panel"),
    worldForm: document.querySelector("#world-form"),
    scenarioSelect: document.querySelector("#scenario-select"),
    createWorldButton: document.querySelector("#create-world-button"),
    resetWorldButton: document.querySelector("#reset-world-button"),
    worldSummary: document.querySelector("#world-summary"),
    worldState: document.querySelector("#world-state"),
    activeScenario: document.querySelector("#active-scenario"),
    activeWorldId: document.querySelector("#active-world-id"),
    taskForm: document.querySelector("#task-form"),
    taskInput: document.querySelector("#task-input"),
    taskHint: document.querySelector("#task-hint"),
    sendButton: document.querySelector("#send-button"),
    newTaskButton: document.querySelector("#new-task-button"),
    mixedPreset: document.querySelector("#mixed-preset"),
    writePreset: document.querySelector("#write-preset"),
    runStatus: document.querySelector("#run-status"),
    resultContent: document.querySelector("#result-content"),
    approvalCard: document.querySelector("#approval-card"),
    approvalToolName: document.querySelector("#approval-tool-name"),
    approvalArguments: document.querySelector("#approval-arguments"),
    approveButton: document.querySelector("#approve-button"),
    trajectoryPanel: document.querySelector("#trajectory-panel"),
    toolTrajectory: document.querySelector("#tool-trajectory"),
    toolCount: document.querySelector("#tool-count"),
};

async function apiFetch(path, { method = "GET", body } = {}) {
    const options = { method, headers: { Accept: "application/json" } };

    if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
    }

    const response = await fetch(path, options);
    const text = await response.text();
    let payload = null;

    if (text) {
        try {
            payload = JSON.parse(text);
        } catch {
            if (!response.ok) {
                throw new Error(`Request failed (${response.status}): ${text}`);
            }

            throw new Error("The server returned an invalid JSON response.");
        }
    }

    if (!response.ok) {
        const detail = payload && payload.detail;
        const message =
            typeof detail === "string"
                ? detail
                : detail
                    ? JSON.stringify(detail)
                    : `Request failed with status ${response.status}.`;
        throw new Error(message);
    }

    return payload;
}

function setBusy(value) {
    busy = value;
    updateControls();
}

function updateControls() {
    const hasWorld = currentWorld !== null;
    const hasTask = elements.taskInput.value.trim().length > 0;

    elements.scenarioSelect.disabled = busy;
    elements.createWorldButton.disabled = busy;
    elements.resetWorldButton.disabled = busy || !hasWorld;
    elements.taskInput.disabled = busy || !hasWorld;
    elements.sendButton.disabled = busy || !hasWorld || !hasTask;
    elements.mixedPreset.disabled = busy;
    elements.writePreset.disabled = busy;
    elements.newTaskButton.disabled = busy || currentRun === null;
    elements.approveButton.disabled = busy;

    elements.taskHint.textContent = hasWorld
        ? "The task runs once against the active world."
        : "Create a demo world to enable execution.";
}

function showError(error) {
    elements.errorPanel.textContent =
        error instanceof Error ? error.message : "An unexpected error occurred.";
    elements.errorPanel.hidden = false;
}

function clearError() {
    elements.errorPanel.textContent = "";
    elements.errorPanel.hidden = true;
}

function renderWorld() {
    if (currentWorld === null) {
        elements.worldSummary.hidden = true;
        elements.worldState.textContent = "Not created";
        elements.createWorldButton.textContent = "Create world";
    } else {
        elements.activeScenario.textContent = currentWorld.scenario;
        elements.activeWorldId.textContent = currentWorld.world_id;
        elements.worldSummary.hidden = false;
        elements.worldState.textContent = "Active";
        elements.createWorldButton.textContent = "Replace world";
    }

    updateControls();
}

function clearRun() {
    currentRun = null;
    elements.runStatus.hidden = true;
    elements.runStatus.textContent = "";
    elements.resultContent.className = "empty-state";
    elements.resultContent.textContent = "No task has been run in this world.";
    elements.approvalCard.hidden = true;
    elements.trajectoryPanel.hidden = true;
    elements.toolTrajectory.replaceChildren();
    updateControls();
}

function showLoading(message) {
    elements.runStatus.hidden = true;
    elements.resultContent.className = "loading-state";
    elements.resultContent.textContent = message;
    elements.approvalCard.hidden = true;
    elements.trajectoryPanel.hidden = true;
}

function statusClass(status) {
    if (["completed", "success"].includes(status)) {
        return "status-success";
    }

    if (status === "approval_required") {
        return "status-approval";
    }

    return "status-error";
}

function renderRun(run) {
  currentRun = run;

  elements.runStatus.textContent = run.status;
  elements.runStatus.className = `status-badge ${statusClass(run.status)}`;
  elements.runStatus.hidden = false;
  elements.resultContent.className = run.final_answer ? "answer-text" : "empty-state";

  if (run.final_answer) {
    renderAnswer(run.final_answer);
  } else {
    elements.resultContent.textContent = "No final answer is available yet.";
  }

  renderApproval(run);
  renderTrajectory(run.tool_executions || []);
  updateControls();
}

function renderAnswer(answer) {
  const parts = answer.split(/(\*\*[^*\n]+\*\*)/g);
  const nodes = parts.map((part) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = part.slice(2, -2);
      return strong;
    }

    return document.createTextNode(part);
  });

  elements.resultContent.replaceChildren(...nodes);
}

function renderApproval(run) {
    const pending = run.status === "approval_required" ? run.pending_approval : null;

    if (!pending) {
        elements.approvalCard.hidden = true;
        return;
    }

    elements.approvalToolName.textContent = pending.tool_name;
    elements.approvalArguments.textContent = JSON.stringify(pending.arguments, null, 2);
    elements.approveButton.textContent = "Approve";
    elements.approvalCard.hidden = false;
}

function renderTrajectory(executions) {
    elements.toolTrajectory.replaceChildren();
    elements.toolCount.textContent = `${executions.length} execution${executions.length === 1 ? "" : "s"}`;
    elements.trajectoryPanel.hidden = executions.length === 0;

    for (const execution of executions) {
        const details = document.createElement("details");
        details.className = `tool-execution ${statusClass(execution.status)}`;

        const summary = document.createElement("summary");
        const indicator = document.createElement("span");
        indicator.className = "tool-indicator";
        indicator.setAttribute("aria-hidden", "true");
        indicator.textContent = execution.status === "success" ? "OK" : "!";

        const name = document.createElement("strong");
        name.textContent = execution.tool_name;

        const outcome = document.createElement("span");
        outcome.className = "tool-outcome";
        outcome.textContent = execution.error_code
            ? `${execution.status} · ${execution.error_code}`
            : execution.status;

        summary.append(indicator, name, outcome);

        const argumentsBlock = document.createElement("pre");
        argumentsBlock.textContent = JSON.stringify(execution.arguments, null, 2);

        details.append(summary, argumentsBlock);
        elements.toolTrajectory.append(details);
    }
}

async function deleteWorldBestEffort(worldId) {
    try {
        await apiFetch(`/worlds/${encodeURIComponent(worldId)}`, { method: "DELETE" });
    } catch (error) {
        console.warn("Could not clean up the previous demo world.", error);
    }
}

elements.worldForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    setBusy(true);

    try {
        if (currentWorld !== null) {
            await deleteWorldBestEffort(currentWorld.world_id);
            currentWorld = null;
            clearRun();
            renderWorld();
        }

        currentWorld = await apiFetch("/worlds", {
            method: "POST",
            body: { scenario: elements.scenarioSelect.value },
        });
        clearRun();
        renderWorld();
    } catch (error) {
        showError(error);
    } finally {
        setBusy(false);
    }
});

elements.resetWorldButton.addEventListener("click", async () => {
    if (currentWorld === null) {
        return;
    }

    clearError();
    setBusy(true);

    try {
        await apiFetch(`/worlds/${encodeURIComponent(currentWorld.world_id)}`, {
            method: "DELETE",
        });
        currentWorld = null;
        clearRun();
        renderWorld();
    } catch (error) {
        showError(error);
    } finally {
        setBusy(false);
    }
});

elements.taskForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (currentWorld === null || !elements.taskInput.value.trim()) {
        return;
    }

    clearError();
    currentRun = null;
    showLoading("Running the agent and tools...");
    setBusy(true);

    try {
        const run = await apiFetch("/agent/runs", {
            method: "POST",
            body: {
                world_id: currentWorld.world_id,
                message: elements.taskInput.value.trim(),
            },
        });
        renderRun(run);
    } catch (error) {
        showError(error);
        elements.resultContent.className = "empty-state";
        elements.resultContent.textContent = "The agent run did not complete.";
    } finally {
        setBusy(false);
    }
});

elements.approveButton.addEventListener("click", async () => {
    if (currentRun === null || currentRun.status !== "approval_required") {
        return;
    }

    clearError();
    elements.approveButton.textContent = "Approving...";
    setBusy(true);

    try {
        const run = await apiFetch(
            `/agent/runs/${encodeURIComponent(currentRun.run_id)}/approve`,
            { method: "POST" },
        );
        renderRun(run);
    } catch (error) {
        showError(error);
        elements.approveButton.textContent = "Approve";
    } finally {
        setBusy(false);
    }
});

elements.mixedPreset.addEventListener("click", () => {
    elements.taskInput.value = MIXED_RAG_TASK;
    elements.taskInput.focus();
    updateControls();
});

elements.writePreset.addEventListener("click", () => {
    elements.taskInput.value = WRITE_APPROVAL_TASK;
    elements.taskInput.focus();
    updateControls();
});

elements.newTaskButton.addEventListener("click", () => {
    elements.taskInput.value = "";
    clearRun();
    elements.taskInput.focus();
});

elements.taskInput.addEventListener("input", updateControls);

renderWorld();
clearRun();
