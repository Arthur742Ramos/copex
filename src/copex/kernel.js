/**
 * Persistent JavaScript REPL kernel for Copex.
 *
 * Protocol: newline-delimited JSON over stdin/stdout.
 *   Request:  { "id": <int>, "code": <string> }
 *             { "id": <int>, "action": "reset" }
 *   Response: { "id": <int>, "result": <string|null>, "error": <string|null>,
 *               "console": <string[]> }
 *
 * State persistence: `var` declarations and unscoped assignments persist
 * across calls.  `let`/`const` are automatically promoted to `var` so
 * that variables survive between executions.
 */

const { createInterface } = require("node:readline");
const vm = require("node:vm");

const TIMEOUT_MS = 30_000;

const _CONTEXT_GLOBALS = {
  setTimeout,
  setInterval,
  clearTimeout,
  clearInterval,
  URL,
  TextEncoder,
  TextDecoder,
  structuredClone,
};

let context = vm.createContext({ console: _makeConsoleProxy([]), ..._CONTEXT_GLOBALS });

function _makeConsoleProxy(buffer) {
  const fmt = (...args) =>
    args
      .map((a) => (typeof a === "string" ? a : JSON.stringify(a) ?? String(a)))
      .join(" ");
  return {
    log: (...args) => buffer.push(fmt(...args)),
    warn: (...args) => buffer.push(`[warn] ${fmt(...args)}`),
    error: (...args) => buffer.push(`[error] ${fmt(...args)}`),
    info: (...args) => buffer.push(`[info] ${fmt(...args)}`),
    debug: (...args) => buffer.push(`[debug] ${fmt(...args)}`),
    dir: (obj) => buffer.push(JSON.stringify(obj, null, 2) ?? String(obj)),
  };
}

function _resetContext() {
  context = vm.createContext({ console: _makeConsoleProxy([]), ..._CONTEXT_GLOBALS });
}

/**
 * Promote top-level `let` and `const` to `var` so declarations persist
 * in the VM context across executions.
 */
function _promoteDeclarations(code) {
  return code.replace(
    /^([ \t]*)(let|const)\s/gm,
    (_, indent, keyword) => `${indent}var `
  );
}

async function _execute(code, consoleBuf) {
  context.console = _makeConsoleProxy(consoleBuf);
  const promoted = _promoteDeclarations(code);

  // 1) Try as a plain expression first (handles `2 + 2`, `x`, etc.)
  try {
    const exprScript = new vm.Script(`(${promoted})`, { filename: "repl.js" });
    const result = exprScript.runInContext(context, { timeout: TIMEOUT_MS });
    return { result: result === undefined ? null : String(result), error: null };
  } catch {
    // Not a valid expression — fall through.
  }

  // 2) Execute as statements.  Wrap in async IIFE to support top-level await
  //    and return the value of the last expression.
  try {
    const script = new vm.Script(promoted, { filename: "repl.js" });
    const result = script.runInContext(context, { timeout: TIMEOUT_MS });
    return { result: result === undefined ? null : String(result), error: null };
  } catch {
    // Sync execution failed — try async wrapper for `await` support.
  }

  try {
    const asyncWrapped = `(async () => {\n${promoted}\n})()`;
    const script = new vm.Script(asyncWrapped, { filename: "repl.js" });
    const promise = script.runInContext(context, { timeout: TIMEOUT_MS });
    const result = await promise;
    return { result: result === undefined ? null : String(result), error: null };
  } catch (err) {
    return { result: null, error: String(err) };
  }
}

const rl = createInterface({ input: process.stdin });

rl.on("line", async (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    process.stdout.write(
      JSON.stringify({ id: null, result: null, error: "invalid JSON", console: [] }) + "\n"
    );
    return;
  }

  const id = msg.id ?? null;

  if (msg.action === "reset") {
    _resetContext();
    process.stdout.write(
      JSON.stringify({ id, result: "context reset", error: null, console: [] }) + "\n"
    );
    return;
  }

  const code = msg.code ?? "";
  if (!code.trim()) {
    process.stdout.write(
      JSON.stringify({ id, result: null, error: "empty code", console: [] }) + "\n"
    );
    return;
  }

  const consoleBuf = [];
  const { result, error } = await _execute(code, consoleBuf);
  process.stdout.write(
    JSON.stringify({ id, result, error, console: consoleBuf }) + "\n"
  );
});

rl.on("close", () => process.exit(0));
