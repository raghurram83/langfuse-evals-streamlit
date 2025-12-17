import React, { useEffect, useMemo, useState } from "react";

type UrlEntry = {
  id: string;
  url: string;
  status: string;
  lastCrawledAt: string;
  chunks?: number;
  crawlDate?: string;
};

type DocumentEntry = {
  id: string;
  name: string;
  sizeMb: number;
  status: string;
  chunks: number;
};

type ConversationStats = {
  total: number;
  callsBot: number;
  callsAgent: number;
  chatsBot: number;
  chatsAgent: number;
};

type ConversationSession = {
  id: string;
  channel: "Call" | "WhatsApp";
  roleMix: string;
  lastMessageAt: string;
  status: string;
};

type ProjectData = {
  urls: UrlEntry[];
  documents: DocumentEntry[];
  conversationStats: ConversationStats;
  sessions: ConversationSession[];
};

type Toast = { id: string; message: string; type?: "info" | "success" | "error" };

const projects = [
  { id: "a", name: "Project A – Wedding Biryani" },
  { id: "b", name: "Project B – Darkins" },
  { id: "c", name: "Project C – MyOperator Demo" },
];

const createMockProjectData = (): ProjectData => ({
  urls: [
    {
      id: crypto.randomUUID(),
      url: "https://example.com/menu",
      status: "Crawled",
      lastCrawledAt: "2024-07-01 10:30",
      chunks: 18,
      crawlDate: "2024-07-01",
    },
    {
      id: crypto.randomUUID(),
      url: "https://example.com/about",
      status: "Ready",
      lastCrawledAt: "2024-06-28 09:10",
      chunks: 7,
      crawlDate: "2024-06-28",
    },
    {
      id: crypto.randomUUID(),
      url: "https://example.com/faq",
      status: "Crawled",
      lastCrawledAt: "2024-06-30 14:05",
      chunks: 12,
      crawlDate: "2024-06-30",
    },
  ],
  documents: [
    { id: crypto.randomUUID(), name: "Menu v2.pdf", sizeMb: 12, status: "Processed", chunks: 25 },
    { id: crypto.randomUUID(), name: "Store-Locations.xlsx", sizeMb: 4, status: "Pending", chunks: 8 },
    { id: crypto.randomUUID(), name: "FAQ.docx", sizeMb: 2, status: "Processed", chunks: 15 },
  ],
  conversationStats: {
    total: 300,
    callsBot: 120,
    callsAgent: 80,
    chatsBot: 60,
    chatsAgent: 40,
  },
  sessions: Array.from({ length: 5 }).map((_, idx) => ({
    id: `SES-${3400 + idx}`,
    channel: idx % 2 === 0 ? "Call" : "WhatsApp",
    roleMix: idx % 2 === 0 ? "Bot/Agent" : "Agent-only",
    lastMessageAt: `2024-07-0${idx + 1} 12:${10 + idx}`,
    status: idx % 2 === 0 ? "Exported" : "Pending",
  })),
});

const AppLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="min-h-screen bg-slate-50 text-slate-900">
    <header className="border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
      <h1 className="text-xl font-semibold text-slate-900">Data Ingestion &amp; Normalisation Layer (MVP)</h1>
      <p className="text-sm text-slate-500">Internal demo • Mocked data &amp; flows only</p>
    </header>
    <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
  </div>
);

const ProjectSelector: React.FC<{
  value: string;
  onChange: (id: string) => void;
}> = ({ value, onChange }) => (
  <div className="mb-4 flex items-center justify-between rounded-lg bg-white p-4 shadow-sm">
    <div>
      <p className="text-xs uppercase tracking-wide text-slate-500">Active project</p>
      <p className="text-lg font-medium text-slate-900">{projects.find((p) => p.id === value)?.name}</p>
    </div>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
    >
      {projects.map((p) => (
        <option key={p.id} value={p.id}>
          {p.name}
        </option>
      ))}
    </select>
  </div>
);

const Tabs: React.FC<{ value: string; onChange: (v: string) => void; options: string[] }> = ({
  value,
  onChange,
  options,
}) => (
  <div className="mb-4 flex gap-2 rounded-lg bg-white p-2 shadow-sm">
    {options.map((opt) => {
      const active = opt === value;
      return (
        <button
          key={opt}
          onClick={() => onChange(opt)}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium ${
            active
              ? "bg-indigo-600 text-white shadow"
              : "text-slate-700 hover:bg-slate-100 hover:text-slate-900"
          }`}
        >
          {opt}
        </button>
      );
    })}
  </div>
);

const HelperText: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p className="text-xs text-slate-500">{children}</p>
);

const Card: React.FC<{ title: string; description?: string; children: React.ReactNode }> = ({
  title,
  description,
  children,
}) => (
  <div className="flex flex-col rounded-lg bg-white p-4 shadow-sm">
    <div className="mb-3">
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      {description && <p className="text-sm text-slate-600">{description}</p>}
    </div>
    {children}
  </div>
);

const Toasts: React.FC<{
  toasts: Toast[];
  onDismiss: (id: string) => void;
}> = ({ toasts, onDismiss }) => (
  <div className="fixed bottom-4 right-4 flex flex-col gap-2">
    {toasts.map((t) => (
      <div
        key={t.id}
        className={`flex items-center gap-3 rounded-md px-4 py-3 text-sm shadow-lg ${
          t.type === "error"
            ? "bg-rose-50 text-rose-700 ring-1 ring-rose-200"
            : "bg-slate-900 text-white ring-1 ring-slate-800"
        }`}
      >
        <span>{t.message}</span>
        <button onClick={() => onDismiss(t.id)} className="text-xs underline">
          Close
        </button>
      </div>
    ))}
  </div>
);

const WebsiteSetupCard: React.FC<{
  urls: UrlEntry[];
  onAddUrl: (url: string) => void;
  onSave: () => void;
}> = ({ urls, onAddUrl, onSave }) => {
  const [draftUrl, setDraftUrl] = useState("");
  const used = urls.length;
  return (
    <Card
      title="Website"
      description="Add your website URLs. We’ll crawl and extract content into the Content Lake."
    >
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm text-slate-600">
          <span className="font-medium text-slate-900">{used}</span> / 30 URLs used
        </div>
        <button
          onClick={() => {
            if (!draftUrl.trim()) return;
            onAddUrl(draftUrl.trim());
            setDraftUrl("");
          }}
          className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-700"
        >
          + Add URL
        </button>
      </div>
      <textarea
        value={draftUrl}
        onChange={(e) => setDraftUrl(e.target.value)}
        placeholder="https://example.com/page"
        className="mb-3 h-20 w-full rounded-md border border-slate-200 px-3 py-2 text-sm shadow-inner focus:border-indigo-500 focus:outline-none"
      />
      <div className="mb-3 overflow-hidden rounded-md border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">URL</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Last Crawled</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {urls.map((u) => (
              <tr key={u.id}>
                <td className="px-3 py-2">{u.url}</td>
                <td className="px-3 py-2">
                  <span className="rounded-full bg-emerald-50 px-2 py-1 text-xs font-medium text-emerald-700">
                    {u.status}
                  </span>
                </td>
                <td className="px-3 py-2 text-slate-600">{u.lastCrawledAt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        onClick={onSave}
        className="mt-auto w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-slate-800"
      >
        Save Website Setup
      </button>
      <HelperText>We only use mock data here. No real ingestion is happening.</HelperText>
    </Card>
  );
};

const DocumentSetupCard: React.FC<{
  documents: DocumentEntry[];
  onAddDocument: (doc: DocumentEntry) => void;
}> = ({ documents, onAddDocument }) => {
  const [showForm, setShowForm] = useState(false);
  const [name, setName] = useState("");
  const [size, setSize] = useState(10);
  const [error, setError] = useState("");

  const handleAdd = () => {
    if (size > 100) {
      setError("File size must be ≤ 100 MB (mock rule).");
      return;
    }
    onAddDocument({
      id: crypto.randomUUID(),
      name: name || "Untitled",
      sizeMb: size,
      status: "Processed",
      chunks: Math.floor(size * 1.5) + 5,
    });
    setName("");
    setSize(10);
    setError("");
    setShowForm(false);
  };

  return (
    <Card
      title="Documents"
      description="Upload documents to extract text into the Content Lake."
    >
      <div className="mb-3 flex items-center justify-between text-sm text-slate-600">
        <span>
          <span className="font-medium text-slate-900">{documents.length}</span> / 10 documents uploaded
        </span>
        <button
          onClick={() => setShowForm((s) => !s)}
          className="rounded-md bg-indigo-50 px-3 py-2 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
        >
          Upload Document (mock)
        </button>
      </div>
      {showForm && (
        <div className="mb-3 space-y-2 rounded-md border border-slate-200 bg-slate-50 p-3">
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Document Name"
              className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
            />
            <select
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
              className="w-40 rounded-md border border-slate-200 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
            >
              {[5, 10, 20, 50, 80, 120].map((v) => (
                <option key={v} value={v}>
                  {v} MB
                </option>
              ))}
            </select>
          </div>
          {error && <p className="text-xs text-rose-600">{error}</p>}
          <div className="flex justify-end gap-2">
            <button
              onClick={() => {
                setShowForm(false);
                setError("");
              }}
              className="rounded-md px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100"
            >
              Cancel
            </button>
            <button
              onClick={handleAdd}
              className="rounded-md bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow hover:bg-slate-800"
            >
              Add
            </button>
          </div>
        </div>
      )}
      <div className="mb-3 overflow-hidden rounded-md border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Name</th>
              <th className="px-3 py-2 text-left">Size (MB)</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Chunks</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {documents.map((d) => (
              <tr key={d.id}>
                <td className="px-3 py-2">{d.name}</td>
                <td className="px-3 py-2">{d.sizeMb}</td>
                <td className="px-3 py-2">{d.status}</td>
                <td className="px-3 py-2">{d.chunks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <HelperText>We only use mock data here. No real ingestion is happening.</HelperText>
    </Card>
  );
};

const ConversationSetupCard: React.FC<{
  stats: ConversationStats;
  onVerify: (ban: string) => void;
  onExportDelta: () => void;
}> = ({ stats, onVerify, onExportDelta }) => {
  const [ban, setBan] = useState("");
  return (
    <Card
      title="Conversations (Calls + WhatsApp)"
      description="Verify your BAN to fetch recent calls and WhatsApp chats."
    >
      <div className="space-y-3">
        <input
          value={ban}
          onChange={(e) => setBan(e.target.value)}
          placeholder="Business Account Number (BAN)"
          className="w-full rounded-md border border-slate-200 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
        />
        <div className="flex gap-2">
          <button
            onClick={() => onVerify(ban)}
            className="flex-1 rounded-md bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow hover:bg-indigo-700"
          >
            Verify &amp; Export Last 300 Conversations
          </button>
          <button
            onClick={onExportDelta}
            className="rounded-md bg-indigo-50 px-3 py-2 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
          >
            Export Delta Conversations
          </button>
        </div>
        <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
          <p className="text-sm font-semibold text-slate-900">Stats (mock)</p>
          <p className="text-sm text-slate-700">
            Calls – Bot: {stats.callsBot} | Agent: {stats.callsAgent}
          </p>
          <p className="text-sm text-slate-700">
            Chats – Bot: {stats.chatsBot} | Agent: {stats.chatsAgent}
          </p>
        </div>
        <HelperText>We only use mock data here. No real ingestion is happening.</HelperText>
      </div>
    </Card>
  );
};

const IngestionSetupTab: React.FC<{
  projectData: ProjectData;
  setProjectData: React.Dispatch<React.SetStateAction<ProjectData>>;
  onNavigateDataManagement: () => void;
  pushToast: (t: Toast) => void;
}> = ({ projectData, setProjectData, onNavigateDataManagement, pushToast }) => {
  const addUrl = (url: string) =>
    setProjectData((prev) => ({
      ...prev,
      urls: [
        ...prev.urls,
        {
          id: crypto.randomUUID(),
          url,
          status: "Ready",
          lastCrawledAt: "—",
          chunks: Math.floor(Math.random() * 15) + 3,
          crawlDate: "—",
        },
      ],
    }));

  const addDoc = (doc: DocumentEntry) => setProjectData((p) => ({ ...p, documents: [...p.documents, doc] }));

  const verifyBan = (ban: string) => {
    if (!ban.trim()) {
      pushToast({ id: crypto.randomUUID(), message: "Enter a BAN to verify (mock).", type: "error" });
      return;
    }
    pushToast({
      id: crypto.randomUUID(),
      message: "BAN verified. 300 conversations exported (mock).",
      type: "success",
    });
  };

  const exportDelta = () =>
    pushToast({ id: crypto.randomUUID(), message: "Fetched delta conversations since last export (mock)." });

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-700">
        Set up where we should fetch data from. This will power your Conversation Lake and Content Lake.
      </p>
      <div className="grid gap-4 md:grid-cols-3">
        <WebsiteSetupCard
          urls={projectData.urls}
          onAddUrl={addUrl}
          onSave={() => pushToast({ id: crypto.randomUUID(), message: "Website setup saved (mock)." })}
        />
        <DocumentSetupCard documents={projectData.documents} onAddDocument={addDoc} />
        <ConversationSetupCard stats={projectData.conversationStats} onVerify={verifyBan} onExportDelta={exportDelta} />
      </div>
      <button
        onClick={onNavigateDataManagement}
        className="mt-2 w-full rounded-md bg-slate-900 px-4 py-3 text-sm font-semibold text-white shadow hover:bg-slate-800"
      >
        Go to Project Screen / Data Management
      </button>
    </div>
  );
};

const WebsiteDataPanel: React.FC<{
  urls: UrlEntry[];
  setUrls: (urls: UrlEntry[]) => void;
  pushToast: (t: Toast) => void;
}> = ({ urls, setUrls, pushToast }) => {
  const [draftUrl, setDraftUrl] = useState("");
  const totalUrls = urls.length;
  const totalChunks = urls.reduce((acc, u) => acc + (u.chunks || 0), 0);

  const recrawl = (id: string) => {
    setUrls(
      urls.map((u) => (u.id === id ? { ...u, status: "Re-crawling..." } : u)),
    );
    setTimeout(() => {
      setUrls(
        urls.map((u) =>
          u.id === id
            ? {
                ...u,
                status: "Crawled",
                lastCrawledAt: new Date().toISOString().slice(0, 16).replace("T", " "),
              }
            : u,
        ),
      );
      pushToast({ id: crypto.randomUUID(), message: "Re-crawl complete (mock)." });
    }, 1200);
  };

  const addUrl = () => {
    if (!draftUrl.trim()) return;
    setUrls([
      ...urls,
      {
        id: crypto.randomUUID(),
        url: draftUrl.trim(),
        status: "Ready",
        lastCrawledAt: "—",
        chunks: Math.floor(Math.random() * 10) + 3,
        crawlDate: new Date().toISOString().slice(0, 10),
      },
    ]);
    setDraftUrl("");
  };

  return (
    <div className="space-y-3 rounded-lg bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-900">
          Total URLs: {totalUrls} | Total Chunks: {totalChunks} (mock)
        </p>
        <div className="flex gap-2">
          <input
            value={draftUrl}
            onChange={(e) => setDraftUrl(e.target.value)}
            placeholder="Add URL"
            className="w-64 rounded-md border border-slate-200 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
          />
          <button
            onClick={addUrl}
            className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-700"
          >
            Add URL
          </button>
        </div>
      </div>
      <div className="overflow-hidden rounded-md border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">URL</th>
              <th className="px-3 py-2 text-left">Crawl Date</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Chunks</th>
              <th className="px-3 py-2 text-left">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {urls.map((u) => (
              <tr key={u.id}>
                <td className="px-3 py-2">{u.url}</td>
                <td className="px-3 py-2">{u.crawlDate || "—"}</td>
                <td className="px-3 py-2">{u.status}</td>
                <td className="px-3 py-2">{u.chunks ?? "—"}</td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => recrawl(u.id)}
                    className="rounded-md bg-indigo-50 px-2 py-1 text-xs font-semibold text-indigo-700 ring-1 ring-indigo-200 hover:bg-indigo-100"
                  >
                    Re-crawl
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const DocumentDataPanel: React.FC<{
  documents: DocumentEntry[];
  setDocuments: (docs: DocumentEntry[]) => void;
}> = ({ documents, setDocuments }) => {
  const totalDocs = documents.length;
  const totalChunks = documents.reduce((acc, d) => acc + d.chunks, 0);

  const removeDoc = (id: string) => setDocuments(documents.filter((d) => d.id !== id));

  return (
    <div className="space-y-3 rounded-lg bg-white p-4 shadow-sm">
      <p className="text-sm font-medium text-slate-900">
        Total Documents: {totalDocs} | Total Chunks: {totalChunks} (mock)
      </p>
      <div className="overflow-hidden rounded-md border border-slate-200">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Name</th>
              <th className="px-3 py-2 text-left">Size</th>
              <th className="px-3 py-2 text-left">Status</th>
              <th className="px-3 py-2 text-left">Chunks</th>
              <th className="px-3 py-2 text-left">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {documents.map((d) => (
              <tr key={d.id}>
                <td className="px-3 py-2">{d.name}</td>
                <td className="px-3 py-2">{d.sizeMb} MB</td>
                <td className="px-3 py-2">{d.status}</td>
                <td className="px-3 py-2">{d.chunks}</td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => removeDoc(d.id)}
                    className="rounded-md bg-rose-50 px-2 py-1 text-xs font-semibold text-rose-700 ring-1 ring-rose-200 hover:bg-rose-100"
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const Bar: React.FC<{ label: string; value: number; total: number; color: string }> = ({
  label,
  value,
  total,
  color,
}) => {
  const width = total === 0 ? 0 : Math.max(5, (value / total) * 100);
  return (
    <div>
      <div className="flex justify-between text-xs text-slate-600">
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div className="mt-1 h-2 rounded-full bg-slate-100">
        <div className="h-2 rounded-full" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

const ConversationDataPanel: React.FC<{
  stats: ConversationStats;
  sessions: ConversationSession[];
  setStats: (s: ConversationStats) => void;
  setSessions: (s: ConversationSession[]) => void;
  pushToast: (t: Toast) => void;
}> = ({ stats, sessions, setStats, pushToast }) => {
  const reexport = () => pushToast({ id: crypto.randomUUID(), message: "Re-exported delta conversations (mock)." });

  const reset = () => {
    if (window.confirm("Reset data for BAN? (mock)")) {
      setStats({ total: 0, callsBot: 0, callsAgent: 0, chatsBot: 0, chatsAgent: 0 });
      pushToast({ id: crypto.randomUUID(), message: "BAN data reset (mock)." });
    }
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-3 md:grid-cols-4">
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <p className="text-xs uppercase text-slate-500">Total Conversations</p>
          <p className="text-2xl font-semibold text-slate-900">{stats.total}</p>
        </div>
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <p className="text-xs uppercase text-slate-500">Calls – Bot</p>
          <p className="text-2xl font-semibold text-slate-900">{stats.callsBot}</p>
        </div>
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <p className="text-xs uppercase text-slate-500">Calls – Agent</p>
          <p className="text-2xl font-semibold text-slate-900">{stats.callsAgent}</p>
        </div>
        <div className="rounded-lg bg-white p-4 shadow-sm">
          <p className="text-xs uppercase text-slate-500">Chats – Bot/Agent</p>
          <p className="text-2xl font-semibold text-slate-900">
            {stats.chatsBot}/{stats.chatsAgent}
          </p>
        </div>
      </div>
      <div className="rounded-lg bg-white p-4 shadow-sm">
        <div className="grid gap-3 md:grid-cols-2">
          <Bar label="Calls – Bot" value={stats.callsBot} total={stats.total} color="#6366F1" />
          <Bar label="Calls – Agent" value={stats.callsAgent} total={stats.total} color="#10B981" />
          <Bar label="Chats – Bot" value={stats.chatsBot} total={stats.total} color="#F59E0B" />
          <Bar label="Chats – Agent" value={stats.chatsAgent} total={stats.total} color="#EF4444" />
        </div>
      </div>
      <div className="flex gap-2">
        <button
          onClick={reexport}
          className="rounded-md bg-indigo-600 px-3 py-2 text-xs font-semibold text-white shadow hover:bg-indigo-700"
        >
          Re-export Delta
        </button>
        <button
          onClick={reset}
          className="rounded-md bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700 ring-1 ring-rose-200 hover:bg-rose-100"
        >
          Reset Data for BAN
        </button>
      </div>
      <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="px-3 py-2 text-left">Session ID</th>
              <th className="px-3 py-2 text-left">Channel</th>
              <th className="px-3 py-2 text-left">Role Mix</th>
              <th className="px-3 py-2 text-left">Last Message At</th>
              <th className="px-3 py-2 text-left">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {sessions.map((s) => (
              <tr key={s.id}>
                <td className="px-3 py-2">{s.id}</td>
                <td className="px-3 py-2">{s.channel}</td>
                <td className="px-3 py-2">{s.roleMix}</td>
                <td className="px-3 py-2">{s.lastMessageAt}</td>
                <td className="px-3 py-2">{s.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const DataManagementTab: React.FC<{
  projectData: ProjectData;
  setProjectData: React.Dispatch<React.SetStateAction<ProjectData>>;
  pushToast: (t: Toast) => void;
}> = ({ projectData, setProjectData, pushToast }) => {
  const [subTab, setSubTab] = useState<"Website" | "Documents" | "Conversations">("Website");

  const subTabButtons = (
    <div className="mb-3 flex gap-2 rounded-lg bg-white p-2 shadow-sm">
      {["Website", "Documents", "Conversations"].map((t) => {
        const active = subTab === t;
        return (
          <button
            key={t}
            onClick={() => setSubTab(t as any)}
            className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${
              active ? "bg-indigo-600 text-white shadow" : "text-slate-700 hover:bg-slate-100"
            }`}
          >
            {t} Data
          </button>
        );
      })}
    </div>
  );

  return (
    <div className="space-y-4">
      {subTabButtons}
      {subTab === "Website" && (
        <WebsiteDataPanel
          urls={projectData.urls}
          setUrls={(urls) => setProjectData((p) => ({ ...p, urls }))}
          pushToast={pushToast}
        />
      )}
      {subTab === "Documents" && (
        <DocumentDataPanel
          documents={projectData.documents}
          setDocuments={(documents) => setProjectData((p) => ({ ...p, documents }))}
        />
      )}
      {subTab === "Conversations" && (
        <ConversationDataPanel
          stats={projectData.conversationStats}
          sessions={projectData.sessions}
          setStats={(stats) => setProjectData((p) => ({ ...p, conversationStats: stats }))}
          setSessions={(sessions) => setProjectData((p) => ({ ...p, sessions }))}
          pushToast={pushToast}
        />
      )}
    </div>
  );
};

const DataIngestionApp: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"Ingestion Setup" | "Data Management">("Ingestion Setup");
  const [selectedProject, setSelectedProject] = useState(projects[0].id);
  const [projectData, setProjectData] = useState<Record<string, ProjectData>>(() => ({
    [projects[0].id]: createMockProjectData(),
  }));
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    setToasts((prev) => prev);
  }, []);

  const currentData = useMemo(() => {
    if (!projectData[selectedProject]) {
      setProjectData((prev) => ({ ...prev, [selectedProject]: createMockProjectData() }));
    }
    return projectData[selectedProject] || createMockProjectData();
  }, [projectData, selectedProject]);

  const pushToast = (toast: Toast) => {
    setToasts((prev) => [...prev, toast]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== toast.id)), 3000);
  };

  const switchProject = (id: string) => {
    setSelectedProject(id);
    setProjectData((prev) => {
      if (!prev[id]) {
        return { ...prev, [id]: createMockProjectData() };
      }
      return prev;
    });
  };

  const setCurrentProjectData = (updater: React.SetStateAction<ProjectData>) => {
    setProjectData((prev) => ({
      ...prev,
      [selectedProject]: typeof updater === "function" ? (updater as any)(currentData) : updater,
    }));
  };

  return (
    <AppLayout>
      <ProjectSelector value={selectedProject} onChange={switchProject} />
      <Tabs value={activeTab} onChange={(v) => setActiveTab(v as any)} options={["Ingestion Setup", "Data Management"]} />
      {activeTab === "Ingestion Setup" ? (
        <IngestionSetupTab
          projectData={currentData}
          setProjectData={setCurrentProjectData}
          onNavigateDataManagement={() => setActiveTab("Data Management")}
          pushToast={pushToast}
        />
      ) : (
        <DataManagementTab projectData={currentData} setProjectData={setCurrentProjectData} pushToast={pushToast} />
      )}
      <Toasts toasts={toasts} onDismiss={(id) => setToasts((prev) => prev.filter((t) => t.id !== id))} />
    </AppLayout>
  );
};

export default DataIngestionApp;
