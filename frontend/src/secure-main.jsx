import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import SecureTransferPanel from "./components/SecureTransferPanel.jsx";


createRoot(document.getElementById("root")).render(
  <StrictMode>
    <div className="min-h-screen text-slate-200">
      <header className="border-b border-slate-800/80 bg-[#070b12]/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4 lg:px-8">
          <div>
            <p className="font-semibold text-white">AegisFlow</p>
            <p className="text-xs text-slate-500">Adaptive Secure Transfer</p>
          </div>
          <a
            href="/"
            className="rounded-xl border border-slate-800 bg-slate-900 px-4 py-2 text-sm text-slate-300 transition hover:border-blue-500/40 hover:text-white"
          >
            Back to Dashboard
          </a>
        </div>
      </header>
      <SecureTransferPanel />
    </div>
  </StrictMode>
);
