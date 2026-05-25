import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart3,
  BriefcaseBusiness,
  CircleDollarSign,
  LineChart,
  KeyRound,
  LogOut,
  RefreshCcw,
  Shield,
  Sparkles,
  Target,
  TrendingUp,
  UserPlus,
  Users,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

const HORIZONS = [
  { value: "short", label: "Kisa", range: "1-5 gun", tone: "Hizli tepki" },
  { value: "medium", label: "Orta", range: "10-20 gun", tone: "Dengeli secim" },
  { value: "long", label: "Uzun", range: "1-3 ay", tone: "Trend odakli" },
];

const NAV_ITEMS = [
  { value: "dashboard", label: "Radar", icon: Sparkles },
  { value: "recommendations", label: "Sinyaller", icon: CircleDollarSign },
  { value: "symbol", label: "Hisse", icon: LineChart },
  { value: "market", label: "Piyasa", icon: BarChart3 },
  { value: "portfolio", label: "Sepet", icon: BriefcaseBusiness },
  { value: "admin", label: "Erisim", icon: Users },
];

function formatDate(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("tr-TR", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

/**
 * Convert a string to plain ASCII uppercase, safe for Turkish-locale browsers.
 *
 * On Turkish-locale systems JavaScript maps 'i' → 'İ' (U+0130, dotted capital I)
 * which breaks ticker lookups stored as plain ASCII.  This helper first strips
 * Turkish-specific characters to their ASCII base, then calls .toUpperCase().
 */
function asciiUpper(text) {
  const turkishMap = {
    "\u0130": "I",  // İ (dotted capital I)
    "\u0131": "I",  // ı (dotless lowercase i)
    "\u015e": "S",  // Ş
    "\u015f": "S",  // ş
    "\u011e": "G",  // Ğ
    "\u011f": "G",  // ğ
    "\u00dc": "U",  // Ü
    "\u00fc": "U",  // ü
    "\u00d6": "O",  // Ö
    "\u00f6": "O",  // ö
    "\u00c7": "C",  // Ç
    "\u00e7": "C",  // ç
  };
  let result = "";
  for (const ch of text.trim()) {
    if (turkishMap[ch] !== undefined) {
      result += turkishMap[ch];
    } else {
      result += ch;
    }
  }
  return result.toUpperCase();
}

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toLocaleString("tr-TR", { maximumFractionDigits: digits });
}

function formatPrice(value) {
  if (value === null || value === undefined) return "-";
  return Number(value).toLocaleString("tr-TR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const sign = Number(value) > 0 ? "+" : "";
  return `${sign}${Number(value).toLocaleString("tr-TR", { maximumFractionDigits: 2 })}%`;
}

function currentHorizon(value) {
  return HORIZONS.find((item) => item.value === value) || HORIZONS[1];
}

function elapsedText(days) {
  if (days === null || days === undefined) return "Olcum yok";
  if (days === 0) return "Bugun uretildi";
  return `${days} gun acik`;
}

function scoreLabel(score) {
  if (score >= 75) return "Yuksek";
  if (score >= 65) return "Izlenebilir";
  return "Sinirda";
}

function returnClass(value) {
  if (value === null || value === undefined) return "flat";
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "flat";
}

async function apiFetch(path, token, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the default error.
    }
    throw new Error(detail);
  }

  return response.status === 204 ? null : response.json();
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin123");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const token = await apiFetch("/auth/login", null, {
        method: "POST",
        body: JSON.stringify({ username_or_email: username, password }),
      });
      const me = await apiFetch("/auth/me", token.access_token);
      onLogin(token.access_token, me);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-screen">
      <section className="login-card">
        <div className="login-orbit">
          <TrendingUp size={34} />
        </div>
        <span className="eyebrow">BIST signal cockpit</span>
        <h1>Tek ekranda sade karar akisi.</h1>
        <p>Guncel sinyalleri, hedefleri ve kullanici erisimini tek platformdan yonet.</p>
        <form onSubmit={submit} className="login-form">
          <label>
            Kullanici
            <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
          </label>
          <label>
            Sifre
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
            />
          </label>
          {error && <div className="error-note">{error}</div>}
          <button className="primary-action" type="submit" disabled={loading}>
            <KeyRound size={18} />
            {loading ? "Kontrol ediliyor" : "Giris Yap"}
          </button>
        </form>
      </section>
    </main>
  );
}

function HorizonSwitch({ horizon, setHorizon }) {
  return (
    <div className="horizon-switch">
      {HORIZONS.map((item) => (
        <button
          key={item.value}
          type="button"
          className={horizon === item.value ? "selected" : ""}
          onClick={() => setHorizon(item.value)}
        >
          <strong>{item.label}</strong>
          <span>{item.range}</span>
        </button>
      ))}
    </div>
  );
}

function MetricPill({ label, value }) {
  return (
    <div className="metric-pill">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SignalCard({ row }) {
  const statusClass = returnClass(row.return_pct);
  return (
    <article className="signal-card">
      <header>
        <div>
          <span className="signal-kind">{row.direction}</span>
          <h3>{row.ticker}</h3>
          <p>{row.name || row.sector || "BIST"}</p>
        </div>
        <div className="score-ring">
          <strong>{formatNumber(row.final_score, 0)}</strong>
          <span>{scoreLabel(row.final_score)}</span>
        </div>
      </header>

      <div className="signal-main">
        <div>
          <span>Su an</span>
          <strong className={statusClass}>{formatPercent(row.return_pct)}</strong>
          <small>{elapsedText(row.days_open)}</small>
        </div>
        <div>
          <span>Vade</span>
          <strong>{currentHorizon(row.horizon).label}</strong>
          <small>{row.horizon_days} gun hedef</small>
        </div>
      </div>

      <div className="price-ladder">
        <div>
          <span>Giris</span>
          <b>{formatPrice(row.entry_price)}</b>
        </div>
        <div>
          <span>Guncel</span>
          <b>{formatPrice(row.current_price)}</b>
        </div>
        <div>
          <span>Hedef</span>
          <b>{formatPrice(row.target_price)}</b>
          <small>{formatPercent(row.target_return_pct)}</small>
        </div>
        <div>
          <span>Stop</span>
          <b>{formatPrice(row.stop_price)}</b>
          <small>{formatPercent(row.stop_return_pct)}</small>
        </div>
      </div>

      <details>
        <summary>Skor ayrintisi</summary>
        <div className="score-grid">
          <span>Trend <b>{formatNumber(row.trend_score)}</b></span>
          <span>Hacim <b>{formatNumber(row.volume_score)}</b></span>
          <span>Momentum <b>{formatNumber(row.relative_strength_score)}</b></span>
          <span>Risk <b>{formatNumber(row.risk_score)}</b></span>
        </div>
        <p>{row.reason}</p>
      </details>
    </article>
  );
}

function EmptyState({ horizon }) {
  const active = currentHorizon(horizon);
  return (
    <div className="empty-panel">
      <Target size={32} />
      <h3>{active.label} vade icin sinyal yok</h3>
      <p>
        {active.label} vade secici skor filtrelerini gecen hisse bulunamadi.
        Sinyal uretmek icin pipeline calistirilmalidir:
      </p>
      <code style={{ display: "block", marginTop: "0.5rem", fontSize: "0.75rem", opacity: 0.7 }}>
        python scripts/run_pipeline.py --horizons short,medium,long
      </code>
    </div>
  );
}

function useRecommendations(token, horizon) {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    apiFetch(`/signals/recommendations?timeframe=1d&horizon=${horizon}&limit=20`, token)
      .then((data) => {
        if (active) setRows(data);
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token, horizon]);

  return { rows, loading, error };
}

function SignalBoard({ token, horizon, compact = false }) {
  const { rows, loading, error } = useRecommendations(token, horizon);

  if (loading) return <div className="loading-panel">Sinyaller hazirlaniyor...</div>;
  if (error) return <div className="empty-panel danger">{error}</div>;
  if (!rows.length) return <EmptyState horizon={horizon} />;

  if (compact) {
    return (
      <div className="signal-strip">
        {rows.slice(0, 3).map((row) => (
          <SignalCard key={row.signal_id} row={row} />
        ))}
      </div>
    );
  }

  return (
    <div className="signal-layout">
      {rows.map((row) => (
        <SignalCard key={row.signal_id} row={row} />
      ))}
    </div>
  );
}

function DashboardPage({ overview, horizon, setHorizon, token }) {
  const active = currentHorizon(horizon);
  return (
    <>
      <section className="command-hero">
        <div className="hero-copy">
          <span className="eyebrow">Canli radar</span>
          <h1>{active.label} vade icin karar ekrani</h1>
          <p>Once en guclu sinyal, sonra takip edilecek alternatifler. Hedef, stop ve guncel performans ayni kartta.</p>
        </div>
        <div className="hero-metrics">
          <MetricPill label="Sembol" value={overview?.counts?.symbols ?? "-"} />
          <MetricPill label="Acik sinyal" value={overview?.counts?.open_signals ?? "-"} />
          <MetricPill label="Vade" value={active.range} />
        </div>
      </section>

      <HorizonSwitch horizon={horizon} setHorizon={setHorizon} />
      <SignalBoard token={token} horizon={horizon} />
    </>
  );
}

function RecommendationsPage({ token, horizon, setHorizon }) {
  return (
    <>
      <section className="page-heading">
        <div>
          <span className="eyebrow">Sinyal arama</span>
          <h1>Vade bazli hisse onerileri</h1>
          <p>Liste, son uretilen secili sepet snapshot'indan beslenir.</p>
        </div>
        <HorizonSwitch horizon={horizon} setHorizon={setHorizon} />
      </section>
      <SignalBoard token={token} horizon={horizon} />
    </>
  );
}

function PortfolioPage({ overview, horizon, setHorizon }) {
  const portfolio = overview?.latest_portfolios?.find((item) => item.horizon === horizon);
  return (
    <>
      <section className="page-heading">
        <div>
          <span className="eyebrow">Sepet kompozisyonu</span>
          <h1>{currentHorizon(horizon).label} vade sepeti</h1>
          <p>{portfolio ? `${formatDate(portfolio.snapshot_time)} tarihinde uretildi` : "Bu vade icin sepet yok"}</p>
        </div>
        <HorizonSwitch horizon={horizon} setHorizon={setHorizon} />
      </section>
      {!portfolio || portfolio.items.length === 0 ? (
        <EmptyState horizon={horizon} />
      ) : (
        <section className="portfolio-mosaic">
          {portfolio.items.map((item) => (
            <article key={`${portfolio.id}-${item.symbol_id}`} className="portfolio-tile">
              <span>#{item.rank}</span>
              <h3>{item.ticker}</h3>
              <b>{formatNumber(item.suggested_weight * 100)}%</b>
              <small>{formatNumber(item.score)} skor</small>
            </article>
          ))}
        </section>
      )}
    </>
  );
}

function SymbolInsightPage({ token }) {
  const [ticker, setTicker] = useState("THYAO");
  const [symbols, setSymbols] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const matches = useMemo(() => {
    return symbols.slice(0, 10);
  }, [symbols]);

  async function searchSymbols(query) {
    const q = asciiUpper(query);
    if (!q) {
      // Load all active symbols when query is empty
      const data = await apiFetch("/symbols?active_only=true&bist100_only=false&limit=500", token);
      setSymbols(data);
      return;
    }
    const data = await apiFetch(`/symbols/search?q=${encodeURIComponent(q)}&active_only=true&limit=25`, token);
    setSymbols(data);
  }

  async function load(event, nextTicker = ticker) {
    event?.preventDefault();
    if (!nextTicker.trim()) return;
    const cleanTicker = asciiUpper(nextTicker);
    setTicker(cleanTicker);
    setLoading(true);
    setError("");
    try {
      const data = await apiFetch(`/symbols/${cleanTicker}/analysis?timeframe=1d`, token);
      setAnalysis(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadWatchlist() {
    const data = await apiFetch("/watchlist", token);
    setWatchlist(data);
  }

  async function addToWatchlist() {
    if (!analysis?.ticker) return;
    await apiFetch("/watchlist", token, {
      method: "POST",
      body: JSON.stringify({ ticker: analysis.ticker }),
    });
    await loadWatchlist();
  }

  async function forceAnalyze(event) {
    event?.preventDefault();
    const cleanTicker = asciiUpper(ticker);
    if (!cleanTicker) return;
    setTicker(cleanTicker);
    setLoading(true);
    setError("");
    try {
      const data = await apiFetch(`/symbols/${cleanTicker}/analyze?timeframe=1d`, token, { method: "POST" });
      setAnalysis(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    searchSymbols("");
    loadWatchlist().catch(() => setWatchlist([]));
    load();
  }, []);

  return (
    <>
      <section className="page-heading">
        <div>
          <span className="eyebrow">Hisse mercegi</span>
          <h1>Bir hisse sec, durumunu oku</h1>
          <p>Teknik gorunum, son sinyal ve karar loglari ayni yerde.</p>
        </div>
        <form className="symbol-search" onSubmit={load}>
          <input value={ticker} onChange={(event) => { setTicker(event.target.value); searchSymbols(event.target.value); }} placeholder="THYAO" />
          <button className="primary-action" type="submit">Incele</button>
          <button className="primary-action" type="button" onClick={forceAnalyze} title="Veri yoksa fiyat çekip teknik göstergeleri hesaplar">
            <RefreshCcw size={16} /> Analiz Et
          </button>
        </form>
      </section>
      <section className="symbol-suggestions">
        {matches.map((symbol) => (
          <button type="button" key={symbol.id} onClick={(event) => load(event, symbol.ticker)}>
            <strong>{symbol.ticker}</strong>
            <span>{symbol.name || symbol.sector || symbol.market}</span>
          </button>
        ))}
      </section>
      {loading && <div className="loading-panel">Hisse inceleniyor...</div>}
      {error && <div className="empty-panel danger">{error}</div>}
      {analysis && (
        <section className="insight-grid">
          <article className="insight-main">
            <span className="eyebrow">{analysis.ticker}</span>
            <h2>{analysis.name || analysis.sector || "BIST"}</h2>
            <p>{analysis.summary}</p>
            <div className="hero-metrics">
              <MetricPill label="Son fiyat" value={formatPrice(analysis.price?.close)} />
              <MetricPill label="5 gun" value={formatPercent(analysis.price?.return_5d)} />
              <MetricPill label="20 gun" value={formatPercent(analysis.price?.return_20d)} />
            </div>
            <button className="primary-action watch-action" type="button" onClick={addToWatchlist}>
              Watchlist'e ekle
            </button>
          </article>
          <article className="admin-card">
            <span className="eyebrow">Teknik skor</span>
            <div className="score-grid">
              <span>Trend <b>{formatNumber(analysis.feature?.trend_score)}</b></span>
              <span>Hacim <b>{formatNumber(analysis.feature?.volume_score)}</b></span>
              <span>Momentum <b>{formatNumber(analysis.feature?.momentum_score)}</b></span>
              <span>RSI <b>{formatNumber(analysis.feature?.rsi)}</b></span>
            </div>
          </article>
          <article className="admin-card">
            <span className="eyebrow">Son sinyaller</span>
            {analysis.latest_signals.length === 0 ? (
              <p>Kayitli sinyal yok.</p>
            ) : (
              analysis.latest_signals.slice(0, 4).map((signal) => (
                <div className="audit-row" key={signal.id}>
                  <b>{signal.direction} · {signal.horizon}</b>
                  <span>{formatDate(signal.signal_time)} · {formatPrice(signal.entry_price)} · skor {formatNumber(signal.final_score)}</span>
                </div>
              ))
            )}
          </article>
          <article className="admin-card">
            <span className="eyebrow">Karar loglari</span>
            {analysis.decision_logs.length === 0 ? (
              <p>Bu hisse icin karar logu yok.</p>
            ) : (
              analysis.decision_logs.slice(0, 6).map((log) => (
                <div className="audit-row" key={log.id}>
                  <b>{log.direction} · {log.horizon}</b>
                  <span>{formatDate(log.decision_time)} · giris {formatPrice(log.entry_price)} · hedef {formatPrice(log.target_price)}</span>
                </div>
              ))
            )}
          </article>
          <article className="admin-card">
            <span className="eyebrow">Watchlist</span>
            {watchlist.length === 0 ? (
              <p>Izleme listende hisse yok.</p>
            ) : (
              watchlist.slice(0, 8).map((item) => (
                <button className="watch-row" type="button" key={item.id} onClick={(event) => load(event, item.ticker)}>
                  <b>{item.ticker}</b>
                  <span>{item.name || item.sector || "BIST"}</span>
                </button>
              ))
            )}
          </article>
        </section>
      )}
    </>
  );
}

function MarketPage({ token }) {
  const [radar, setRadar] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    apiFetch("/market/radar?timeframe=1d", token)
      .then((data) => {
        if (active) setRadar(data);
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token]);

  if (loading) return <div className="loading-panel">Piyasa radari hazirlaniyor...</div>;
  if (error) return <div className="empty-panel danger">{error}</div>;

  return (
    <>
      <section className="page-heading">
        <div>
          <span className="eyebrow">Endeks radari</span>
          <h1>Turk borsasi piyasa modu</h1>
          <p>{radar?.breadth_summary}</p>
        </div>
        <MetricPill label="Risk modu" value={radar?.risk_mode || "-"} />
      </section>
      <section className="portfolio-mosaic">
        {radar?.indices.map((index) => (
          <article className="portfolio-tile index-tile" key={index.ticker}>
            <span>{index.ticker}</span>
            <h3>{index.label}</h3>
            <b>{formatPercent(index.return_5d)}</b>
            <small>{index.summary}</small>
          </article>
        ))}
      </section>
    </>
  );
}

function AdminUsers({ token, me }) {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ username: "", email: "", password: "", role: "viewer", full_name: "" });
  const [message, setMessage] = useState("");
  const isAdmin = me?.role === "admin";

  async function loadUsers() {
    if (!isAdmin) return;
    const data = await apiFetch("/users", token);
    setUsers(data);
  }

  useEffect(() => {
    loadUsers().catch((err) => setMessage(err.message));
  }, [token, isAdmin]);

  async function createUser(event) {
    event.preventDefault();
    setMessage("");
    try {
      await apiFetch("/users", token, { method: "POST", body: JSON.stringify(form) });
      setForm({ username: "", email: "", password: "", role: "viewer", full_name: "" });
      await loadUsers();
      setMessage("Kullanici olusturuldu.");
    } catch (err) {
      setMessage(err.message);
    }
  }

  async function toggleUser(user) {
    setMessage("");
    try {
      await apiFetch(`/users/${user.id}`, token, {
        method: "PATCH",
        body: JSON.stringify({ is_active: !user.is_active }),
      });
      await loadUsers();
    } catch (err) {
      setMessage(err.message);
    }
  }

  if (!isAdmin) {
    return (
      <section className="empty-panel">
        <Shield size={32} />
        <h3>Admin yetkisi gerekiyor</h3>
        <p>Kullanici olusturma ve erisim kapatma sadece admin rolune acik.</p>
      </section>
    );
  }

  return (
    <section className="admin-grid">
      <form className="admin-card" onSubmit={createUser}>
        <span className="eyebrow">Yeni erisim</span>
        <h2>Kullanici olustur</h2>
        <input placeholder="kullanici adi" value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} />
        <input placeholder="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
        <input placeholder="sifre" type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
        <select value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value })}>
          <option value="viewer">viewer</option>
          <option value="analyst">analyst</option>
          <option value="trial">trial</option>
          <option value="admin">admin</option>
        </select>
        <button className="primary-action" type="submit">
          <UserPlus size={17} />
          Olustur
        </button>
        {message && <p className="admin-message">{message}</p>}
      </form>

      <section className="admin-card user-directory">
        <span className="eyebrow">Erisim listesi</span>
        <h2>Kullanicilar</h2>
        {users.map((user) => (
          <div className="user-row" key={user.id}>
            <div>
              <strong>{user.username}</strong>
              <span>{user.email} · {user.role}</span>
            </div>
            <button type="button" onClick={() => toggleUser(user)} className={user.is_active ? "soft danger" : "soft"}>
              {user.is_active ? "Kapat" : "Ac"}
            </button>
          </div>
        ))}
      </section>
    </section>
  );
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("bist_token") || "");
  const [me, setMe] = useState(null);
  const [overview, setOverview] = useState(null);
  const [horizon, setHorizon] = useState("medium");
  const [activePage, setActivePage] = useState("dashboard");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(Boolean(token));

  const activeNav = useMemo(() => NAV_ITEMS.find((item) => item.value === activePage), [activePage]);

  async function loadAll(nextToken = token) {
    setError("");
    setLoading(true);
    try {
      const [meData, overviewData] = await Promise.all([
        apiFetch("/auth/me", nextToken),
        apiFetch("/dashboard/overview?timeframe=1d", nextToken),
      ]);
      setMe(meData);
      setOverview(overviewData);
    } catch (err) {
      setError(err.message);
      if (err.message.includes("Invalid") || err.message.includes("authenticated")) logout();
    } finally {
      setLoading(false);
    }
  }

  function handleLogin(nextToken, user) {
    localStorage.setItem("bist_token", nextToken);
    setToken(nextToken);
    setMe(user);
    loadAll(nextToken);
  }

  function logout() {
    localStorage.removeItem("bist_token");
    setToken("");
    setMe(null);
    setOverview(null);
  }

  useEffect(() => {
    if (token) loadAll(token);
  }, []);

  if (!token) return <Login onLogin={handleLogin} />;

  return (
    <main className="terminal-shell">
      <header className="app-header">
        <button className="brand-button" type="button" onClick={() => setActivePage("dashboard")}>
          <TrendingUp size={22} />
          <span>BIST Signal</span>
        </button>
        <nav className="top-nav">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                type="button"
                key={item.value}
                className={activePage === item.value ? "active" : ""}
                onClick={() => setActivePage(item.value)}
              >
                <Icon size={17} />
                {item.label}
              </button>
            );
          })}
        </nav>
        <div className="header-actions">
          <button type="button" className="soft" onClick={() => loadAll()}>
            <RefreshCcw size={16} />
            Yenile
          </button>
          <span className="user-chip">{me?.username || "user"} · {me?.role || "-"}</span>
          <button type="button" className="icon-action" onClick={logout} title="Cikis">
            <LogOut size={18} />
          </button>
        </div>
      </header>

      <section className="content-shell">
        <div className="mobile-title">
          <span>{activeNav?.label}</span>
        </div>
        {error && <div className="empty-panel danger">{error}</div>}
        {loading && <div className="loading-panel">Veriler yenileniyor...</div>}

        {activePage === "dashboard" && (
          <DashboardPage overview={overview} horizon={horizon} setHorizon={setHorizon} token={token} />
        )}
        {activePage === "recommendations" && (
          <RecommendationsPage token={token} horizon={horizon} setHorizon={setHorizon} />
        )}
        {activePage === "symbol" && <SymbolInsightPage token={token} />}
        {activePage === "market" && <MarketPage token={token} />}
        {activePage === "portfolio" && <PortfolioPage overview={overview} horizon={horizon} setHorizon={setHorizon} />}
        {activePage === "admin" && <AdminUsers token={token} me={me} />}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
