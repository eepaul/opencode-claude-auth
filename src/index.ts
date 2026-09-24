import type { Plugin as V1Plugin } from "@opencode-ai/plugin"
import type { Plugin as V2 } from "@opencode/plugin"
import crypto from "node:crypto"
import { getUserAgent } from "./model-config.ts"
import { readAllClaudeAccounts, type ClaudeAccount } from "./keychain.ts"
import { initLogger, log } from "./logger.ts"
import { fetchWithRetry } from "./http.ts"
import {
  addExcludedBeta,
  getExcludedBetas,
  getModelBetas,
  getNextBetaToExclude,
  isLongContextError,
  LONG_CONTEXT_BETAS,
} from "./betas.ts"
import {
  SYSTEM_IDENTITY,
  transformBody,
  transformResponseStream,
} from "./transforms.ts"
import {
  getCachedCredentials,
  getCredentialsWithBackoff,
  getActiveRefreshFailureKind,
  reloadCredentialsFromSource,
  forceRefreshActiveAccount,
  getActiveAccount,
  syncAuthJson,
  initAccounts,
  setActiveAccountSource,
  loadPersistedAccountSource,
  saveAccountSource,
  refreshAccountsList,
  refreshIfNeeded,
  type ClaudeCredentials,
} from "./credentials.ts"

export {
  addExcludedBeta,
  getExcludedBetas,
  getModelBetas,
  getNextBetaToExclude,
  isLongContextError,
  LONG_CONTEXT_BETAS,
} from "./betas.ts"
export { resetExcludedBetas } from "./betas.ts"
export { fetchWithRetry, type FetchFn } from "./http.ts"
export {
  stripToolPrefix,
  SYSTEM_IDENTITY,
  transformBody,
  transformResponseStream,
} from "./transforms.ts"
export {
  getCachedCredentials,
  syncAuthJson,
  refreshAccountsList,
  type ClaudeCredentials,
} from "./credentials.ts"
export {
  buildBillingHeaderValue,
  computeCch,
  computeVersionSuffix,
  extractFirstUserMessageText,
} from "./signing.ts"

function getStainlessHeaders(): Record<string, string> {
  return {
    "x-stainless-arch": process.arch === "arm64" ? "arm64" : process.arch,
    "x-stainless-lang": "js",
    "x-stainless-os":
      process.platform === "darwin" ? "MacOS" : process.platform,
    "x-stainless-package-version": "0.81.0",
    "x-stainless-retry-count": "0",
    "x-stainless-runtime": "node",
    "x-stainless-runtime-version": process.version,
    "x-stainless-timeout": "600",
  }
}

function buildRequestUrl(input: RequestInfo | URL): string | URL {
  const raw =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url

  const url = new URL(raw)
  if (url.pathname === "/v1/messages" && !url.searchParams.has("beta")) {
    url.searchParams.set("beta", "true")
  }

  return typeof input === "string" ? url.toString() : url
}

// Stable per-process session ID, matching Claude Code's X-Claude-Code-Session-Id
const sessionId = crypto.randomUUID()

export function buildRequestHeaders(
  input: RequestInfo | URL,
  init: RequestInit,
  accessToken: string,
  modelId = "unknown",
  excludedBetas?: Set<string>,
): Headers {
  const headers = new Headers()

  if (input instanceof Request) {
    input.headers.forEach((value, key) => {
      headers.set(key, value)
    })
  }

  if (init.headers instanceof Headers) {
    init.headers.forEach((value, key) => {
      headers.set(key, value)
    })
  } else if (Array.isArray(init.headers)) {
    for (const [key, value] of init.headers) {
      if (typeof value !== "undefined") {
        headers.set(key, String(value))
      }
    }
  } else if (init.headers) {
    for (const [key, value] of Object.entries(init.headers)) {
      if (typeof value !== "undefined") {
        headers.set(key, String(value))
      }
    }
  }

  const modelBetas = getModelBetas(modelId, excludedBetas)
  const incomingBeta = headers.get("anthropic-beta") ?? ""
  const mergedBetas = [
    ...new Set([
      ...modelBetas,
      ...incomingBeta
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    ]),
  ]

  headers.set("authorization", `Bearer ${accessToken}`)
  headers.set("anthropic-version", "2023-06-01")
  headers.set("anthropic-beta", mergedBetas.join(","))
  headers.set("anthropic-dangerous-direct-browser-access", "true")
  headers.set("x-app", "cli")
  headers.set("user-agent", getUserAgent())
  headers.set("x-client-request-id", crypto.randomUUID())
  headers.set("X-Claude-Code-Session-Id", sessionId)
  for (const [key, value] of Object.entries(getStainlessHeaders())) {
    if (!headers.has(key)) headers.set(key, value)
  }
  headers.delete("x-api-key")

  return headers
}

const SYNC_INTERVAL = 5 * 60 * 1000 // 5 minutes
const PROACTIVE_REFRESH_THRESHOLD_MS = 60 * 60 * 1000 // 1 hour before expiry

/** A request whose auth, headers and body have been rewritten for Claude. */
export interface ClaudeRequest {
  input: RequestInfo | URL
  requestInit: RequestInit
  requestUrl: string | URL
  body: BodyInit | null | undefined
  headers: Headers
  modelId: string
  excluded: Set<string>
  accessToken: string
}

/**
 * Resolves credentials and rewrites the request for the Claude API. Returns a
 * synthetic Response when the request must not be sent (refresh rate-limited),
 * and throws when no credentials are available at all.
 */
export async function prepareClaudeRequest(
  input: RequestInfo | URL,
  requestInit: RequestInit,
): Promise<ClaudeRequest | Response> {
  let latest = await getCachedCredentials()
  if (!latest) {
    // A transient refresh rate-limit must not surface as a hard error.
    // Wait (bounded, abort-aware) for our cooldown to clear or for a
    // sibling OpenCode instance / the claude CLI to write a fresh
    // token to the shared store.
    latest = await getCredentialsWithBackoff({
      signal: requestInit.signal ?? undefined,
    })
  }
  if (!latest) {
    if (getActiveRefreshFailureKind() === "transient") {
      // Retryable: let OpenCode/the AI SDK back off and retry rather
      // than telling the user to re-authenticate for a passing
      // rate-limit that the refresh token would otherwise survive.
      log("fetch_credentials_transient_exhausted", {
        modelId: "unknown",
      })
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "overloaded_error",
            message: "Claude token refresh is rate-limited; retry shortly.",
          },
        }),
        {
          status: 429,
          headers: {
            "content-type": "application/json",
            "retry-after": "5",
          },
        },
      )
    }
    log("fetch_no_credentials", { modelId: "unknown" })
    throw new Error(
      "Claude Code credentials are unavailable or expired. Run `claude` to refresh them.",
    )
  }

  const bodyStr =
    typeof requestInit.body === "string" ? requestInit.body : undefined
  let modelId = "unknown"
  if (bodyStr) {
    try {
      modelId = (JSON.parse(bodyStr) as { model?: string }).model ?? "unknown"
    } catch {}
  }

  log("fetch_credentials", {
    modelId,
    accessToken: latest.accessToken,
    expiresAt: latest.expiresAt,
  })

  // Get excluded betas for this model (from previous failed requests)
  const excluded = getExcludedBetas(modelId)
  const requestUrl = buildRequestUrl(input)
  const headers = buildRequestHeaders(
    input,
    requestInit,
    latest.accessToken,
    modelId,
    excluded,
  )
  const body = transformBody(requestInit.body)

  const headerKeys: string[] = []
  headers.forEach((_, key) => {
    headerKeys.push(key)
  })
  const betas = (headers.get("anthropic-beta") ?? "").split(",").filter(Boolean)
  log("fetch_headers_built", { headerKeys, betas, modelId })

  return {
    input,
    requestInit,
    requestUrl,
    body,
    headers,
    modelId,
    excluded,
    accessToken: latest.accessToken,
  }
}

/**
 * Recovers from auth, rate-limit and long-context failures by re-issuing the
 * prepared request, then rewrites the response stream for OpenCode.
 */
export async function completeClaudeResponse(
  request: ClaudeRequest,
  initialResponse: Response,
): Promise<Response> {
  const { input, requestInit, requestUrl, body, modelId, excluded } = request
  let response = initialResponse

  // Recover from a rejected token: first by adopting credentials
  // rotated externally (cswap switching accounts, the claude CLI,
  // another OpenCode instance), then by forcing an OAuth refresh
  // when the store still holds the token that was just rejected.
  //
  // Most cases resolve on the first attempt: a cold or unreadable
  // store yields null from the reload (reloadCredentialsFromSource
  // rejects anything expiring within 60s), so the force refresh runs
  // immediately. The second attempt covers the narrower race where
  // the reload returns a valid-looking token that a concurrent
  // writer has itself just rotated again.
  //
  // The cap is the real bound. Against a store being rotated on
  // every read, every candidate differs from tokenInUse, so the
  // no-progress break never fires and the cap alone stops the loop.
  // The break is the fast path out of the common cases, not the
  // guarantee of termination.
  //
  // tokenInUse deliberately tracks only the last token tried rather
  // than the set of all of them: a store cycling A->B->A wastes one
  // request on the second attempt, which is a better trade than
  // threading extra state through a loop whose whole virtue is a
  // hard ceiling of three API calls.
  const MAX_AUTH_RECOVERY_ATTEMPTS = 2
  let tokenInUse = request.accessToken

  for (
    let attempt = 0;
    response.status === 401 && attempt < MAX_AUTH_RECOVERY_ATTEMPTS;
    attempt++
  ) {
    let candidate: ClaudeCredentials | null = null
    // reloadCredentialsFromSource already catches its own source
    // read and returns null, so this is unreachable today. It stays
    // because the guarantee worth keeping is that no reload failure
    // turns a well-formed 401 into an exception thrown out of
    // fetch() — degrading to the original response beats crashing
    // the request. It logs so a future reload that does throw is
    // diagnosable rather than silently null-coalesced.
    try {
      candidate = reloadCredentialsFromSource()
    } catch (err) {
      log("auth_recovery_reload_threw", {
        modelId,
        attempt: attempt + 1,
        error: err instanceof Error ? err.message : String(err),
      })
    }

    if (!candidate || candidate.accessToken === tokenInUse) {
      try {
        candidate = await forceRefreshActiveAccount()
      } catch (err) {
        // A rejected refresh and a refresh that returned null are
        // different operator-facing diagnoses; auth_recovery_
        // exhausted below collapses them, so record this one here.
        log("auth_recovery_force_refresh_threw", {
          modelId,
          attempt: attempt + 1,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }

    // Re-checked, not copy-pasted: the guard above decides whether
    // to force a refresh, this one decides whether that refresh
    // actually produced a token worth retrying with.
    if (!candidate || candidate.accessToken === tokenInUse) {
      log("auth_recovery_exhausted", {
        modelId,
        attempt: attempt + 1,
      })
      break
    }

    tokenInUse = candidate.accessToken
    log("auth_recovery_retry", { modelId, attempt: attempt + 1 })
    response = await fetchWithRetry(requestUrl, {
      ...requestInit,
      body,
      headers: buildRequestHeaders(
        input,
        requestInit,
        tokenInUse,
        modelId,
        excluded,
      ),
    })
  }

  // An external switch — cswap rotating off an exhausted account —
  // leaves this session on the old token until the 30s credential
  // cache expires. Re-read once so a rate limit that has already
  // been resolved elsewhere is not surfaced. A changed token is the
  // signal that a switch happened; when nothing changed this costs
  // one source read and no retry.
  //
  // Ordered AFTER the 401 recovery loop, and that is a real
  // dependency, not incidental sequencing: it must compare against
  // the token the loop last tried, so a 401 recovered into a 429 is
  // measured against the recovered token rather than the rejected
  // one. Only half of this is compiler-enforced — hoisting the block
  // above `let tokenInUse` is a TDZ error, but moving it between
  // that declaration and the loop still compiles and still passes,
  // while silently comparing against a stale token on the
  // 401 -> retry -> 429 path.
  //
  // Ordered before the long-context beta loop deliberately. A
  // long-context 429 is a header problem, not an account one, so it
  // rotates no token and falls through here untouched. In the rare
  // case a switch lands on the same 429, this spends one retry that
  // comes back with the same long-context error and the beta loop
  // then handles it off the fresh response — one wasted request,
  // same outcome.
  if (response.status === 429) {
    let rotated: ClaudeCredentials | null = null
    // Unreachable today for the same reason as the 401 loop's
    // reload catch: reloadCredentialsFromSource swallows its own
    // source read and returns null. Kept, and logged, on the same
    // grounds — no reload failure should turn a readable 429 into
    // an exception thrown out of fetch(), and a future reload that
    // does throw should be diagnosable rather than silently
    // coalesced to "nothing rotated".
    try {
      rotated = reloadCredentialsFromSource()
    } catch (err) {
      log("rate_limit_reload_threw", {
        modelId,
        error: err instanceof Error ? err.message : String(err),
      })
    }

    if (rotated && rotated.accessToken !== tokenInUse) {
      // Named for what was observed, not for what it implies. A
      // changed token is not proof of an account switch: a routine
      // refresh of this same exhausted account by another instance
      // or the claude CLI changes the token too, and that retry hits
      // the same quota. Accepted cost — one request — but the log
      // must not tell a quota investigation "we switched accounts"
      // when all it saw was a different token.
      log("rate_limit_token_changed", { modelId })
      tokenInUse = rotated.accessToken
      response = await fetchWithRetry(requestUrl, {
        ...requestInit,
        body,
        headers: buildRequestHeaders(
          input,
          requestInit,
          tokenInUse,
          modelId,
          excluded,
        ),
      })
      // Whether rotating resolved the limit is the question this
      // whole block exists to answer, so record it outright rather
      // than leaving success to be inferred from the absence of a
      // fetch_error_response line.
      log("rate_limit_retry_response", {
        modelId,
        status: response.status,
      })
    }
  }

  // Check for long-context beta errors and retry with betas excluded
  // Try up to LONG_CONTEXT_BETAS.length times, excluding one more beta each time
  for (let attempt = 0; attempt < LONG_CONTEXT_BETAS.length; attempt++) {
    if (response.status !== 400 && response.status !== 429) {
      break
    }

    const cloned = response.clone()
    const responseBody = await cloned.text()

    if (!isLongContextError(responseBody)) {
      break
    }

    const betaToExclude = getNextBetaToExclude(modelId)
    if (!betaToExclude) {
      break // All long-context betas already excluded
    }

    addExcludedBeta(modelId, betaToExclude)
    log("fetch_beta_excluded", {
      modelId,
      excludedBeta: betaToExclude,
    })

    // Rebuild headers without the excluded beta and retry
    // Falls back to tokenInUse, not the prepared token: after a 401
    // recovery the latter is the token the API already rejected.
    const currentCreds = await getCachedCredentials()
    const retryToken = currentCreds?.accessToken ?? tokenInUse
    const newExcluded = getExcludedBetas(modelId)
    const newHeaders = buildRequestHeaders(
      input,
      requestInit,
      retryToken,
      modelId,
      newExcluded,
    )

    response = await fetchWithRetry(requestUrl, {
      ...requestInit,
      body,
      headers: newHeaders,
    })
  }

  // Record non-200 responses without writing over OpenCode's terminal UI.
  if (!response.ok) {
    const status = response.status
    const cloned = response.clone()
    cloned
      .text()
      .then((errorBody) => {
        let message = errorBody
        try {
          const parsed = JSON.parse(errorBody) as {
            error?: { type?: string; message?: string }
          }
          message = parsed.error?.message ?? parsed.error?.type ?? errorBody
        } catch {}
        log("fetch_error_response", { status, modelId, message })
      })
      .catch(() => {})
  }

  // A 401 that survived recovery carries an error body, not an SSE
  // stream. Deciding here rather than from a flag set mid-flight
  // makes the retried and non-retried paths behave identically.
  return response.status === 401 ? response : transformResponseStream(response)
}

async function claudeFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const requestInit = init ?? {}
  const prepared = await prepareClaudeRequest(input, requestInit)
  if (prepared instanceof Response) return prepared

  const response = await fetchWithRetry(prepared.requestUrl, {
    ...requestInit,
    body: prepared.body,
    headers: prepared.headers,
  })

  log("fetch_response", {
    status: response.status,
    modelId: prepared.modelId,
    retryAttempt: 0,
  })

  return completeClaudeResponse(prepared, response)
}

interface ClaudeAuthState {
  accounts: ClaudeAccount[]
  defaultAccountSource: string | null
  stop: () => void
}

/**
 * Loads Claude Code accounts, selects the persisted one and starts the
 * proactive refresh timer. Returns null when credentials cannot be read.
 */
async function initClaudeAuth(
  onCredentials: (creds: ClaudeCredentials) => void,
): Promise<ClaudeAuthState | null> {
  initLogger()

  let accounts: ClaudeAccount[] = []
  try {
    accounts = readAllClaudeAccounts()
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err)
    log("plugin_init_error", { error })
    console.warn(
      "opencode-claude-auth: Failed to read Claude Code credentials:",
      error,
    )
    return null
  }

  initAccounts(accounts)

  const defaultAccountSource = accounts[0]?.source ?? null

  if (accounts.length === 0) {
    log("plugin_init_no_accounts", { reason: "no credentials found" })
    console.warn(
      "opencode-claude-auth: No Claude Code credentials found. Running in API key mode with transform hook enabled.",
    )
    return { accounts, defaultAccountSource, stop() {} }
  }

  const persistedSource = loadPersistedAccountSource()
  const defaultAccount =
    (persistedSource && accounts.find((a) => a.source === persistedSource)) ||
    accounts[0]

  setActiveAccountSource(defaultAccount.source)

  log("plugin_init", {
    accountCount: accounts.length,
    sources: accounts.map((a) => a.source),
    activeSource: defaultAccount.source,
  })

  const initialCreds = await getCachedCredentials()
  if (initialCreds) {
    onCredentials(initialCreds)
  } else {
    console.warn(
      "opencode-claude-auth: Claude credentials are expired and could not be refreshed. Run `claude` to re-authenticate.",
    )
  }

  // Keep credentials synced and proactively refresh before expiry.
  // refreshIfNeeded() always resolves the currently ACTIVE account
  // (via getActiveAccount() internally) — not a closure-captured account
  // list — so this stays correct across account switches. Passing
  // PROACTIVE_REFRESH_THRESHOLD_MS (1 hour) means it triggers a real
  // OAuth refresh once the token is within that window of expiry, and
  // simply returns the untouched credentials otherwise (no-op refresh).
  // This prevents the "run `claude` to re-authenticate" message from
  // appearing mid-session when the token silently expires.
  let proactiveRefreshWarned = false
  const syncTimer = setInterval(async () => {
    try {
      const account = getActiveAccount()
      log("proactive_refresh_check", {
        source: account?.source ?? null,
        expiresAt: account?.credentials?.expiresAt ?? null,
        thresholdMs: PROACTIVE_REFRESH_THRESHOLD_MS,
      })

      const creds = await refreshIfNeeded(
        undefined,
        PROACTIVE_REFRESH_THRESHOLD_MS,
      )
      if (creds) {
        onCredentials(creds)
        if (proactiveRefreshWarned) {
          log("proactive_refresh_recovered", { source: account?.source })
        }
        proactiveRefreshWarned = false
      } else {
        log("proactive_refresh_failed", { source: account?.source })
        // Only warn once per outage — otherwise this fires every
        // SYNC_INTERVAL (5 min) for as long as refresh keeps failing.
        if (!proactiveRefreshWarned) {
          proactiveRefreshWarned = true
          console.warn(
            "opencode-claude-auth: Proactive token refresh failed. Run `claude` to re-authenticate.",
          )
        }
      }
    } catch {
      // Non-fatal
    }
  }, SYNC_INTERVAL)
  syncTimer.unref()

  return {
    accounts,
    defaultAccountSource,
    stop: () => clearInterval(syncTimer),
  }
}

/** Resolves an authorize() account selection to a known account. */
function resolveChosenAccount(
  accounts: ClaudeAccount[],
  selected: string | undefined,
): ClaudeAccount {
  const latestAccounts = refreshAccountsList()
  const source = selected ?? latestAccounts[0]?.source ?? accounts[0].source
  return (
    latestAccounts.find((a) => a.source === source) ??
    accounts.find((a) => a.source === source) ??
    latestAccounts[0] ??
    accounts[0]
  )
}

function describeAccountSource(account: ClaudeAccount): string {
  return account.source === "file"
    ? `credentials file (${account.configDir ?? "~/.claude"}/.credentials.json)`
    : `macOS Keychain (${account.source})`
}

// ---------------------------------------------------------------------------
// V1 plugin (OpenCode 1.x): auth loader with a custom fetch.
// ---------------------------------------------------------------------------

const plugin: V1Plugin = async () => {
  const state = await initClaudeAuth(syncAuthJson)
  if (!state) return {}
  const { accounts, defaultAccountSource } = state

  return {
    "experimental.chat.system.transform": async (input, output) => {
      if (input.model?.providerID !== "anthropic") {
        return
      }

      const hasIdentityPrefix = output.system.some((entry) =>
        entry.includes(SYSTEM_IDENTITY),
      )
      if (!hasIdentityPrefix) {
        output.system.unshift(SYSTEM_IDENTITY)
      }
    },
    auth: {
      provider: "anthropic",
      async loader(getAuth, provider) {
        const auth = await getAuth()
        log("auth_loader_called", { authType: auth.type })
        if (auth.type !== "oauth") {
          log("auth_loader_skipped", {
            authType: auth.type,
            reason: "auth type is not oauth",
          })
          return {}
        }

        for (const model of Object.values(provider.models)) {
          model.cost = {
            input: 0,
            output: 0,
            cache: { read: 0, write: 0 },
          }
        }

        log("auth_loader_ready", {
          modelCount: Object.keys(provider.models).length,
        })

        return {
          apiKey: "",
          baseURL: "https://api.anthropic.com/v1",
          fetch: claudeFetch,
        }
      },
      methods: [
        {
          type: "oauth",
          label: "Switch Claude Code account",

          get prompts() {
            const currentAccounts = refreshAccountsList()
            const currentSource =
              loadPersistedAccountSource() ?? defaultAccountSource
            if (currentAccounts.length <= 1) return []
            return [
              {
                type: "select" as const,
                key: "account",
                message: "Select which Claude Code account to use:",
                options: currentAccounts.map((a) => ({
                  label: a.label,
                  value: a.source,
                  hint: a.source === currentSource ? "active" : undefined,
                })),
              },
            ]
          },

          async authorize(inputs) {
            const chosen = resolveChosenAccount(accounts, inputs?.account)

            setActiveAccountSource(chosen.source)
            const creds = (await getCachedCredentials()) ?? chosen.credentials

            syncAuthJson(creds)
            saveAccountSource(chosen.source)

            return {
              url: "",
              instructions: `Using ${chosen.label} — credentials loaded from ${describeAccountSource(chosen)}.`,
              method: "auto",
              async callback() {
                return {
                  type: "success",
                  provider: "anthropic",
                  access: creds.accessToken,
                  refresh: creds.refreshToken,
                  expires: creds.expiresAt,
                }
              },
            }
          },
        },
      ],
    },
  }
}

// ---------------------------------------------------------------------------
// V2 plugin (OpenCode 2.x): integration method plus session HTTP hooks.
// ---------------------------------------------------------------------------

export const PLUGIN_ID = "opencode-claude-auth"
const PROVIDER_ID = "anthropic"
const OAUTH_METHOD_ID = "claude-code"
const REQUEST_ID_HEADER = "x-client-request-id"
// Requests whose response never arrives (network error, abort) would
// otherwise leave their prepared state behind forever.
const PENDING_REQUEST_TTL_MS = 10 * 60 * 1000

type V2Context = V2.Context

function toOAuthCredential(creds: ClaudeCredentials) {
  return {
    type: "oauth" as const,
    methodID: OAUTH_METHOD_ID as never,
    access: creds.accessToken,
    refresh: creds.refreshToken,
    expires: creds.expiresAt,
  }
}

/**
 * Whether requests to the anthropic provider should use Claude Code OAuth.
 * An API key or environment connection is left untouched, matching V1's
 * `auth.type !== "oauth"` check.
 */
async function isOAuthConnection(ctx: V2Context): Promise<boolean> {
  const connection = await ctx.integration.connection.active(PROVIDER_ID)
  return connection?.type === "credential" && connection.method === "oauth"
}

async function setupV2(ctx: V2Context): Promise<V2.Cleanup | void> {
  const state = await initClaudeAuth(() => {})
  if (!state) return
  const { accounts, defaultAccountSource } = state

  const hasAccounts = accounts.length > 0
  let oauthActive = false
  try {
    oauthActive = hasAccounts && (await isOAuthConnection(ctx))
  } catch (err) {
    log("plugin_v2_connection_error", {
      error: err instanceof Error ? err.message : String(err),
    })
  }

  // V1's experimental.chat.system.transform ran for every model request, so
  // register the identity prefix on each request kind.
  const injectIdentity = (event: { system: Array<{ text: string }> }) => {
    const hasIdentityPrefix = event.system.some((part) =>
      part.text.includes(SYSTEM_IDENTITY),
    )
    if (!hasIdentityPrefix) {
      event.system.unshift({ type: "text", text: SYSTEM_IDENTITY } as never)
    }
  }
  const scope = { providerID: PROVIDER_ID }
  await ctx.session.hook("context", injectIdentity, scope)
  await ctx.session.hook("compaction", injectIdentity, scope)
  await ctx.session.hook("generate", injectIdentity, scope)
  await ctx.session.hook("title", injectIdentity, scope)

  if (!hasAccounts) return state.stop

  // Subscription usage is not billed per token.
  await ctx.model.transform((editor) => {
    if (!oauthActive) return
    for (const model of editor.list(PROVIDER_ID)) {
      editor.update(PROVIDER_ID, String(model.id), (m) => {
        m.cost = []
      })
    }
  })

  const accountOptions = accounts.map((a) => ({
    value: a.source,
    label: a.label,
    description: a.source === defaultAccountSource ? "default" : undefined,
  }))
  await ctx.integration.transform((editor) => {
    editor.method.update({
      integrationID: PROVIDER_ID,
      method: {
        id: OAUTH_METHOD_ID,
        type: "oauth",
        label: "Claude Code account",
        form:
          accountOptions.length > 1
            ? [
                {
                  type: "string",
                  key: "account",
                  title: "Select which Claude Code account to use:",
                  options: accountOptions,
                  default:
                    loadPersistedAccountSource() ??
                    defaultAccountSource ??
                    undefined,
                },
              ]
            : undefined,
      },
      async authorize(answer) {
        const selected =
          typeof answer.account === "string" ? answer.account : undefined
        const chosen = resolveChosenAccount(accounts, selected)

        setActiveAccountSource(chosen.source)
        const creds = (await getCachedCredentials()) ?? chosen.credentials
        saveAccountSource(chosen.source)

        return {
          url: "",
          instructions: `Using ${chosen.label} — credentials loaded from ${describeAccountSource(chosen)}.`,
          mode: "auto",
          callback: Promise.resolve(toOAuthCredential(creds)),
        }
      },
      async refresh(credential) {
        const creds = await getCachedCredentials()
        return creds ? toOAuthCredential(creds) : credential
      },
      label() {
        return getActiveAccount()?.label
      },
    })
  })

  // V1 wrote Claude credentials into auth.json at startup so the provider
  // came up connected without a login step. Do the equivalent through the
  // integration API, but never replace a connection the user chose.
  void (async () => {
    try {
      if (await ctx.integration.connection.active(PROVIDER_ID)) return
      await ctx.integration.oauth.connect({
        integrationID: PROVIDER_ID,
        methodID: OAUTH_METHOD_ID,
      } as never)
      log("plugin_v2_connected", { integrationID: PROVIDER_ID })
    } catch (err) {
      log("plugin_v2_connect_error", {
        error: err instanceof Error ? err.message : String(err),
      })
    }
  })()

  const pending = new Map<string, { request: ClaudeRequest; at: number }>()
  const synthetic = new Map<string, { response: Response; at: number }>()
  const sweep = () => {
    const cutoff = Date.now() - PENDING_REQUEST_TTL_MS
    for (const [id, entry] of pending) if (entry.at < cutoff) pending.delete(id)
    for (const [id, entry] of synthetic)
      if (entry.at < cutoff) synthetic.delete(id)
  }

  await ctx.session.hook(
    "http.request",
    async (event) => {
      const active = await isOAuthConnection(ctx)
      if (active !== oauthActive) {
        oauthActive = active
        void ctx.model.reload().catch(() => {})
      }
      if (!active) {
        log("auth_loader_skipped", { reason: "connection is not oauth" })
        return
      }

      const original = event.request
      const headers = new Headers(original.headers)
      // The body is rewritten below, so the host's length no longer holds.
      headers.delete("content-length")
      const requestInit: RequestInit = {
        method: original.method,
        headers,
        body: original.body ? await original.clone().text() : undefined,
        signal: original.signal,
      }

      sweep()
      const prepared = await prepareClaudeRequest(original.url, requestInit)
      if (prepared instanceof Response) {
        // A hook cannot answer a request itself; send it unauthenticated
        // and substitute the retryable response when it comes back.
        const id = crypto.randomUUID()
        headers.set(REQUEST_ID_HEADER, id)
        headers.delete("authorization")
        headers.delete("x-api-key")
        synthetic.set(id, { response: prepared, at: Date.now() })
        event.request = new Request(original.url, { ...requestInit, headers })
        return
      }

      const id = prepared.headers.get(REQUEST_ID_HEADER)!
      pending.set(id, { request: prepared, at: Date.now() })
      event.request = new Request(prepared.requestUrl, {
        ...requestInit,
        body: prepared.body,
        headers: prepared.headers,
      })
    },
    scope,
  )

  await ctx.session.hook(
    "http.response",
    async (event) => {
      const id = event.request.headers.get(REQUEST_ID_HEADER)
      if (!id) return

      const fallback = synthetic.get(id)
      if (fallback) {
        synthetic.delete(id)
        await event.response.body?.cancel().catch(() => {})
        event.response = fallback.response
        return
      }

      const entry = pending.get(id)
      if (!entry) return
      pending.delete(id)

      log("fetch_response", {
        status: event.response.status,
        modelId: entry.request.modelId,
        retryAttempt: 0,
      })
      event.response = await completeClaudeResponse(
        entry.request,
        event.response,
      )
    },
    scope,
  )

  return () => {
    state.stop()
    pending.clear()
    synthetic.clear()
  }
}

export const ClaudeAuthPlugin = plugin

export const ClaudeAuthPluginV2 = {
  id: PLUGIN_ID,
  setup: setupV2,
} satisfies V2.Plugin

// One default export serves both hosts: V2 calls setup(), V1 (OpenCode
// 1.18.29+) calls server().
export default {
  ...ClaudeAuthPluginV2,
  server: plugin,
}
