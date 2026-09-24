import {
  CheckCircle2,
  Download,
  Eye,
  FileKey2,
  FileSearch,
  LockKeyhole,
  ShieldCheck,
  UploadCloud,
} from "lucide-react";
import { useEffect, useState } from "react";


const API_BASE = "http://127.0.0.1:8000";


function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}


function levelClass(level) {
  switch (String(level).toUpperCase()) {
    case "LOW":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
    case "MEDIUM":
      return "border-blue-500/30 bg-blue-500/10 text-blue-300";
    case "HIGH":
      return "border-amber-500/30 bg-amber-500/10 text-amber-300";
    case "CRITICAL":
      return "border-red-500/30 bg-red-500/10 text-red-300";
    default:
      return "border-slate-700 bg-slate-800 text-slate-300";
  }
}


function readableMode(mode) {
  if (!mode) return "—";
  const parts = [];
  if (mode.includes("local_openclip_vision")) parts.push("OpenCLIP Vision");
  if (mode.includes("local_document_text")) parts.push("Document Text");
  if (mode.includes("plaintext")) parts.push("Plaintext");
  if (mode.includes("metadata")) parts.push("Metadata");
  if (parts.length) return parts.join(" + ");
  if (mode.includes("fallback")) return "Metadata/Text Fallback";
  return mode;
}


function readableVisionStatus(status) {
  if (!status || status === "not_applicable") return "Not applicable";
  if (status.startsWith("active:openclip:")) return "Active (OpenCLIP local)";
  if (status.startsWith("active:")) return "Active (local model)";
  if (status === "disabled") return "Disabled";
  if (status === "pillow_not_installed") return "Unavailable (Pillow missing)";
  if (status.startsWith("model_unavailable:")) return "Unavailable (vision model/dependency)";
  if (status.startsWith("image_analysis_failed:")) return "Image analysis failed";
  return status;
}

function readableVisionGate(status) {
  if (!status) return "—";
  const matched = status.match(/gate=([^;]+)/);
  const gate = matched?.[1];
  if (!gate) return "—";
  if (gate.startsWith("accepted:")) {
    return `Accepted: ${gate.split(":")[1]?.toUpperCase() ?? "UNKNOWN"}`;
  }
  if (gate.startsWith("rejected_weak_sensitive:")) {
    return `Rejected weak ${gate.split(":")[1]?.toUpperCase() ?? "sensitive"} match`;
  }
  if (gate === "rejected_weak_iot") return "Rejected weak IoT match";
  if (gate === "no_scores") return "No visual scores";
  return gate;
}


export default function SecureTransferPanel() {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [lastProtected, setLastProtected] = useState(null);
  const [decryptFile, setDecryptFile] = useState(null);
  const [hermesStatus, setHermesStatus] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [recipientPublicKey, setRecipientPublicKey] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadCapabilities() {
      try {
        const response = await fetch(`${API_BASE}/api/secure/capabilities`);
        if (!response.ok) return;
        const payload = await response.json();
        if (!cancelled) setCapabilities(payload);
      } catch {
        // The main error surface is reserved for user-triggered operations.
      }
    }

    void loadCapabilities();
    return () => {
      cancelled = true;
    };
  }, []);

  async function analyzeFile() {
    if (!file) return;
    setBusy(true);
    setError("");
    setAnalysis(null);

    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_BASE}/api/secure/analyze`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Analysis failed.");
      }
      setAnalysis(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function protectFile() {
    if (!file) return;
    setBusy(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);
      if (recipientPublicKey.trim()) {
        form.append("recipient_public_key", recipientPublicKey.trim());
      }
      const response = await fetch(`${API_BASE}/api/secure/protect`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        let message = "Protection failed.";
        try {
          const payload = await response.json();
          message = payload.detail ?? message;
        } catch {
          // Keep fallback message for non-JSON errors.
        }
        throw new Error(message);
      }

      const manifestHeader = response.headers.get("X-AegisFlow-Manifest");
      const manifest = manifestHeader ? JSON.parse(manifestHeader) : null;
      const blob = await response.blob();
      const protectedName = manifest?.protected_filename ?? `${file.name}.aegis`;
      downloadBlob(blob, protectedName);
      setLastProtected(manifest);
      if (manifest?.analysis) {
        setAnalysis(manifest.analysis);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function unprotectFile() {
    if (!decryptFile) return;
    setBusy(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", decryptFile);
      const response = await fetch(`${API_BASE}/api/secure/unprotect`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        let message = "Unable to decrypt file.";
        try {
          const payload = await response.json();
          message = payload.detail ?? message;
        } catch {
          // Keep fallback message.
        }
        throw new Error(message);
      }

      const disposition = response.headers.get("Content-Disposition") ?? "";
      const matched = disposition.match(/filename="([^"]+)"/);
      const filename = matched?.[1] ?? "recovered.bin";
      downloadBlob(await response.blob(), filename);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const context = analysis?.content_context;
  const decision = analysis?.decision;
  const profile = analysis?.security_profile;

  async function checkHermesReadiness() {
    try {
      const response = await fetch(`${API_BASE}/api/integrations/hermes/v1/readiness`);
      if (!response.ok) throw new Error("Hermes readiness check failed.");
      setHermesStatus(await response.json());
    } catch (err) {
      setHermesStatus({ ready: false, error: err.message });
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-5 py-10 lg:px-8">
      <div className="mb-8">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.22em] text-blue-400">
          Adaptive Enforcement
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-white">
          AegisFlow Secure Transfer
        </h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
          Select a file. AegisFlow automatically detects its context, combines it with
          current network intelligence, selects LOW–CRITICAL protection, then encrypts
          the file using the resolved cryptographic profile. Images can be interpreted
          with the optional local vision model, so a public poster is not treated like
          private content merely because it is an image.
        </p>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section className="rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
          <div className="mb-5 flex items-center gap-3">
            <UploadCloud className="h-5 w-5 text-blue-400" />
            <div>
              <h2 className="font-semibold text-white">Automatic Protection</h2>
              <p className="text-xs text-slate-500">No manual content profile required</p>
            </div>
          </div>

          <label className="block rounded-2xl border border-dashed border-slate-700 bg-slate-900/40 p-8 text-center transition hover:border-blue-500/50">
            <input
              type="file"
              className="hidden"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null);
                setAnalysis(null);
                setLastProtected(null);
              }}
            />
            <FileKey2 className="mx-auto mb-3 h-9 w-9 text-slate-500" />
            <p className="font-medium text-slate-200">
              {file ? file.name : "Choose a file to protect"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              {file ? `${(file.size / 1024).toFixed(1)} KiB` : "Prototype limit: 25 MiB"}
            </p>
          </label>

          <details className="mt-5 rounded-xl border border-slate-800 bg-slate-900/30 p-4">
            <summary className="cursor-pointer text-sm font-medium text-slate-300">
              Recipient / multi-device options
            </summary>
            <div className="mt-4 space-y-3">
              <p className="text-xs leading-relaxed text-slate-500">
                Leave the field empty to protect for this AegisFlow installation. To send
                to another AegisFlow device, paste that device&apos;s X25519 public key.
              </p>
              <textarea
                rows={3}
                value={recipientPublicKey}
                onChange={(event) => setRecipientPublicKey(event.target.value)}
                placeholder="Optional remote recipient public key (Base64)"
                className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 font-mono text-xs text-slate-300 outline-none focus:border-blue-500/50"
              />
              {capabilities?.recipient_public_key && (
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <p className="text-xs text-slate-500">This device public key</p>
                  <p className="mt-1 break-all font-mono text-xs text-slate-300">
                    {capabilities.recipient_public_key}
                  </p>
                </div>
              )}
            </div>
          </details>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <button
              type="button"
              disabled={!file || busy}
              onClick={analyzeFile}
              className="flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-200 transition hover:border-blue-500/40 disabled:opacity-40"
            >
              <FileSearch className="h-4 w-4" />
              {busy ? "Analyzing..." : "Analyze Automatically"}
            </button>
            <button
              type="button"
              disabled={!file || busy}
              onClick={protectFile}
              className="flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-blue-500 disabled:opacity-40"
            >
              <LockKeyhole className="h-4 w-4" />
              Protect & Download
            </button>
          </div>

          {lastProtected && (
            <div className="mt-5 flex items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4">
              <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
              <div>
                <p className="text-sm font-medium text-emerald-300">File protected successfully</p>
                <p className="mt-1 text-xs text-slate-400">
                  Downloaded as {lastProtected.protected_filename}
                </p>
              </div>
            </div>
          )}
        </section>

        <section className="rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
          <div className="mb-5 flex items-center gap-3">
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
            <div>
              <h2 className="font-semibold text-white">Automatic Decision</h2>
              <p className="text-xs text-slate-500">Context → policy AI → crypto profile</p>
            </div>
          </div>

          {!analysis ? (
            <div className="flex min-h-64 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/30 p-8 text-center text-sm text-slate-500">
              Analyze or protect a file to see the detected context and selected security profile.
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <InfoCard label="Detected Context" value={context?.category?.toUpperCase() ?? "—"} />
                <InfoCard
                  label="Context Confidence"
                  value={`${((context?.confidence ?? 0) * 100).toFixed(1)}% (heuristic)`}
                />
                <InfoCard label="Network Threat" value={`${((analysis?.network_context?.threat_score ?? 0) * 100).toFixed(1)}%`} />
                <InfoCard label="Context Source" value={analysis?.network_context?.source ?? "—"} />
                <InfoCard label="Content Intelligence" value={readableMode(context?.analysis_mode)} />
                <InfoCard label="Image Vision" value={readableVisionStatus(context?.vision_status)} />
                {context?.detected_mime?.startsWith("image/") && (
                  <InfoCard label="Vision Security Gate" value={readableVisionGate(context?.vision_status)} />
                )}
                {context?.document_status && context.document_status !== "not_applicable" && (
                  <InfoCard
                    label="Document Intelligence"
                    value={`${String(context?.document_kind ?? "document").toUpperCase()} · ${context.document_status}`}
                  />
                )}
              </div>

              {context?.detected_mime?.startsWith("image/") && (
                <div className="flex items-start gap-3 rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
                  <Eye className="mt-0.5 h-5 w-5 shrink-0 text-blue-400" />
                  <p className="text-xs leading-relaxed text-slate-400">
                    Image classification uses local OpenCLIP visual semantics when available. Sensitive
                    visual matches must also pass a conservative security gate before they can escalate
                    the policy. Context Confidence is a heuristic score, not a measured model accuracy.
                  </p>
                </div>
              )}

              <div className={`rounded-xl border p-4 ${levelClass(decision?.policy)}`}>
                <p className="text-xs uppercase tracking-wider opacity-70">Selected Policy</p>
                <p className="mt-1 text-2xl font-semibold">{decision?.policy ?? "—"}</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Enforced Cryptography
                </p>
                <div className="space-y-2 text-sm">
                  <Pair label="Data cipher" value={profile?.data_cipher} />
                  <Pair label="Key agreement" value={profile?.key_agreement} />
                  <Pair label="Key derivation" value={profile?.kdf} />
                  <Pair label="Session key" value={profile ? `${profile.session_key_bits} bit` : "—"} />
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
                <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Why this context was detected
                </p>
                <div className="flex flex-wrap gap-2">
                  {(context?.signals ?? []).map((signal) => (
                    <span key={signal} className="rounded-full border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-300">
                      {signal}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
        <div className="mb-5 flex items-center gap-3">
          <Download className="h-5 w-5 text-violet-400" />
          <div>
            <h2 className="font-semibold text-white">Recover Protected File</h2>
            <p className="text-xs text-slate-500">
              Local prototype receiver using this AegisFlow installation&apos;s X25519 private key
            </p>
          </div>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row">
          <label className="flex flex-1 items-center rounded-xl border border-slate-700 bg-slate-900 px-4 py-3 text-sm text-slate-300">
            <input
              type="file"
              accept=".aegis"
              className="hidden"
              onChange={(event) => setDecryptFile(event.target.files?.[0] ?? null)}
            />
            {decryptFile ? decryptFile.name : "Choose .aegis file"}
          </label>
          <button
            type="button"
            disabled={!decryptFile || busy}
            onClick={unprotectFile}
            className="rounded-xl border border-violet-500/30 bg-violet-500/10 px-5 py-3 text-sm font-semibold text-violet-300 disabled:opacity-40"
          >
            Decrypt & Download Original
          </button>
        </div>
      </section>

      <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-950/70 p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-400">
              Hermes Integration
            </p>
            <h2 className="mt-1 font-semibold text-white">MCP adapter prepared</h2>
            <p className="mt-1 max-w-2xl text-xs leading-relaxed text-slate-500">
              Hermes can orchestrate AegisFlow through a restricted MCP tool surface while
              AegisFlow remains responsible for classification, policy decisions, and encryption.
            </p>
          </div>
          <button
            type="button"
            onClick={checkHermesReadiness}
            className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-2.5 text-sm font-semibold text-emerald-300"
          >
            Check Hermes Readiness
          </button>
        </div>
        {hermesStatus && (
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <InfoCard label="Adapter Ready" value={hermesStatus.ready ? "YES" : "NO"} />
            <InfoCard label="Monitoring" value={hermesStatus.monitoring ? "ACTIVE" : "STOPPED"} />
            <InfoCard
              label="Backend"
              value={hermesStatus.error ? `ERROR: ${hermesStatus.error}` : "READY"}
            />
          </div>
        )}
      </section>

      <p className="mt-5 text-xs leading-relaxed text-slate-500">
        Scope: Secure Transfer protects files explicitly submitted to AegisFlow. It does not
        intercept or re-encrypt arbitrary Chrome, WhatsApp, banking-app, or other third-party
        traffic. Those applications continue to use their own TLS/security protocols.
      </p>
    </div>
  );
}


function InfoCard({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="mt-2 break-words text-sm font-medium text-slate-200">{value ?? "—"}</p>
    </div>
  );
}


function Pair({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-800/70 pb-2">
      <span className="text-slate-500">{label}</span>
      <span className="font-mono text-slate-200">{value ?? "—"}</span>
    </div>
  );
}
