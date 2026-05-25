
// BIST100 AI Dashboard Logic

// Store the full loaded data so vade switches don't need re-fetch
let _dashboardData = null;
let _currentMode = 'KISA';

document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    setupNavigation();

    // Theme
    setupTheme();

    // Data
    fetchData();

    // Vade Selection
    document.querySelectorAll('.vade-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.vade-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            _currentMode = btn.getAttribute('data-vade') || 'KISA';
            applyMode();
        });
    });

    document.getElementById('refresh-btn').addEventListener('click', fetchData);
});

function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.page-section');
    const pageTitle = document.getElementById('page-title');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();

            // Activate Link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            // Show Section
            const targetId = link.getAttribute('data-target');
            sections.forEach(section => {
                if (section.id === targetId) {
                    section.style.display = 'block';
                    // Trigger animation (optional)
                    section.style.opacity = 0;
                    setTimeout(() => section.style.opacity = 1, 10);
                } else {
                    section.style.display = 'none';
                }
            });

            // Update Title
            const titleMap = {
                'dashboard': 'Piyasa Genel Bakış',
                'signals': 'Tüm Sinyaller',
                'portfolio': 'Portföy Yönetimi',
                'settings': 'Sistem Ayarları'
            };
            pageTitle.textContent = titleMap[targetId];
        });
    });
}

function setupTheme() {
    const toggle = document.getElementById('theme-switch');
    const body = document.body;

    // Check local storage
    if (localStorage.getItem('theme') === 'light') {
        body.classList.add('light-mode');
        body.classList.remove('dark-mode');
        toggle.checked = false;
    }

    toggle.addEventListener('change', () => {
        if (toggle.checked) {
            body.classList.add('dark-mode');
            body.classList.remove('light-mode');
            localStorage.setItem('theme', 'dark');
        } else {
            body.classList.add('light-mode');
            body.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
        }
    });
}

async function fetchData() {
    try {
        const response = await fetch('dashboard_data.json');
        if (!response.ok) throw new Error('Veri yüklenemedi');

        const raw = await response.json();
        _dashboardData = raw;
        applyMode();
    } catch (error) {
        console.error('Error:', error);
        // On local, fallback to dummy data if json missing
        _dashboardData = null;
        updateDashboard(getDummyData());
    }
}

/**
 * Extract mode-specific data from the loaded JSON.
 * Supports two formats:
 *   - New (combined): { modes: { KISA: {...}, ORTA: {...}, UZUN: {...} }, active_mode: "..." }
 *   - Old (flat):     { scan_date: "...", buy_signals: [...], ... }
 */
function getModeData(mode) {
    if (!_dashboardData) return null;
    // New combined format
    if (_dashboardData.modes && _dashboardData.modes[mode]) {
        return _dashboardData.modes[mode];
    }
    // Old flat format – if the stored mode matches, return as-is; otherwise null
    if (_dashboardData.mode === mode) {
        return _dashboardData;
    }
    // Flat format with different mode – no data for this vade
    return null;
}

/**
 * Apply the currently active mode's data to the dashboard.
 */
function applyMode() {
    const modeData = getModeData(_currentMode);

    if (modeData) {
        updateDashboard(modeData);
    } else if (_dashboardData) {
        // We have data but not for this mode – show a helpful message
        document.getElementById('last-update').textContent =
            `Son Güncelleme: ${_dashboardData.scan_date || 'bilinmiyor'} — ${_currentMode} verisi henüz taranmadı`;
        document.getElementById('total-scanned').textContent = '-';
        document.getElementById('buy-count').textContent = '-';
        document.getElementById('sell-count').textContent = '-';
        document.getElementById('market-volatility').textContent = 'Veri yok';

        const tbody = document.querySelector('#signals-table tbody');
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--text-secondary);">
            <i class="fa-solid fa-circle-exclamation"></i> ${_currentMode} vade için henüz tarama yapılmadı.<br>
            <small>python scripts/run_daily_scan.py --mode ${_currentMode} --lookback 200</small>
        </td></tr>`;

        document.getElementById('signal-details').innerHTML = '';
    } else {
        updateDashboard(getDummyData());
    }
}

function updateDashboard(data) {
    // 1. Update Header
    document.getElementById('last-update').textContent = `Son Güncelleme: ${data.scan_date}`;

    // 2. Update Stats
    animateValue('total-scanned', data.total_scanned);
    animateValue('buy-count', data.buy_count);
    animateValue('sell-count', data.sell_count);
    document.getElementById('market-volatility').textContent = data.market_volatility || 'Normal';

    // 3. Populate Signals Table
    const tbody = document.querySelector('#signals-table tbody');
    tbody.innerHTML = '';

    data.buy_signals.slice(0, 10).forEach(signal => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight: 600;">${signal.symbol}</td>
            <td>${signal.current_price.toFixed(2)} ₺</td>
            <td class="signal-buy"><i class="fa-solid fa-arrow-up"></i> AL</td>
            <td>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px;">
                        <div style="width: ${signal.probability * 100}%; height: 100%; background: var(--accent-green); border-radius: 3px;"></div>
                    </div>
                    ${(signal.probability * 100).toFixed(0)}%
                </div>
            </td>
            <td class="target-col">${signal.target_price ? signal.target_price.toFixed(2) + ' ₺' : '-'}</td>
            <td class="horizon-col">${signal.horizon_days ? signal.horizon_days + ' Gün' : '-'}</td>
            <td>${signal.rsi.toFixed(0)}</td>
             <td>
                ${getSentimentBadge(signal.sentiment_score)}
                <div style="font-size: 0.7rem; color: var(--text-secondary);">${signal.history_info || ''}</div>
            </td>
            <td><span class="badge">Aktif</span></td>
        `;
        tbody.appendChild(row);
    });

    // 4. Update Chart
    updateChart(data);

    // 5. Populate Details
    const detailsContainer = document.getElementById('signal-details');
    detailsContainer.innerHTML = '';

    [...data.buy_signals, ...data.sell_signals].forEach(signal => {
        const div = document.createElement('div');
        div.className = `detail-item ${signal.signal.toLowerCase()}`;
        div.innerHTML = `
            <div class="detail-header">
                <span class="detail-symbol">${signal.symbol}</span>
                <span class="${signal.signal === 'BUY' ? 'signal-buy' : 'signal-sell'}">${signal.signal}</span>
            </div>
            <div class="detail-reason">${signal.reason}</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-secondary);">
                RSI: ${signal.rsi.toFixed(0)} | Vol: ${signal.volatility.toFixed(4)} <br>
                Sentiment: ${getSentimentBadge(signal.sentiment_score)}
            </div>
        `;
        detailsContainer.appendChild(div);
    });

    // 6. Populate Portfolio
    if (data.portfolio) {
        document.getElementById('total-equity').textContent = formatCurrency(data.portfolio.total_equity);
        document.getElementById('daily-pnl').textContent = `${formatCurrency(data.portfolio.daily_pnl)} (%${data.portfolio.daily_pnl_pct})`;
        document.getElementById('daily-pnl').className = data.portfolio.daily_pnl >= 0 ? 'pnl-pos' : 'pnl-neg';

        const portBody = document.querySelector('#portfolio-table tbody');
        portBody.innerHTML = '';
        data.portfolio.holdings.forEach(pos => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-weight: 600;">${pos.symbol}</td>
                <td>${pos.quantity}</td>
                <td>${pos.avg_price.toFixed(2)}</td>
                <td>${pos.current_price.toFixed(2)}</td>
                <td class="${pos.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">${formatCurrency(pos.pnl)}</td>
                <td class="${pos.pnl >= 0 ? 'pnl-pos' : 'pnl-neg'}">%${pos.pnl_pct.toFixed(2)}</td>
             `;
            portBody.appendChild(row);
        });
    }

    // 7. Populate Config
    if (data.config) {
        const configGrid = document.getElementById('config-grid');
        configGrid.innerHTML = '';
        Object.entries(data.config).forEach(([key, value]) => {
            const div = document.createElement('div');
            div.className = 'setting-item';
            div.innerHTML = `
                <div class="setting-label">${formatKey(key)}</div>
                <div class="setting-value">${value}</div>
            `;
            configGrid.appendChild(div);
        });
    }
}

function getSentimentBadge(score) {
    if (!score) return '<span style="color: #94a3b8">Nötr</span>';
    if (score > 0.5) return '<span style="color: #10b981">Pozitif 🔥</span>';
    if (score > 0) return '<span style="color: #34d399">Hafif Pozitif</span>';
    if (score < -0.5) return '<span style="color: #ef4444">Negatif 🔻</span>';
    return '<span style="color: #f87171">Hafif Negatif</span>';
}

function formatCurrency(val) {
    return new Intl.NumberFormat('tr-TR', { style: 'currency', currency: 'TRY' }).format(val);
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function updateChart(data) {
    const ctx = document.getElementById('marketChart').getContext('2d');

    // Calculate simple distribution
    const buy = data.buy_count;
    const sell = data.sell_count;
    const hold = data.hold_count;

    if (window.marketChartInstance) {
        window.marketChartInstance.destroy();
    }

    window.marketChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['AL', 'SAT', 'BEKLE'],
            datasets: [{
                data: [buy, sell, hold],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(148, 163, 184, 0.2)'
                ],
                borderColor: 'transparent',
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

function animateValue(id, value) {
    const obj = document.getElementById(id);
    obj.textContent = value;
}

function getDummyData() {
    return {
        "scan_date": "2025-01-01 (Demo)",
        "total_scanned": 100,
        "buy_count": 5,
        "sell_count": 3,
        "hold_count": 92,
        "market_volatility": "Düşük",
        "buy_signals": [
            {
                "symbol": "THYAO",
                "current_price": 285.50,
                "signal": "BUY",
                "probability": 0.85,
                "rsi": 45,
                "volatility": 0.012,
                "sentiment_score": 0.8,
                "reason": "Yapay zeka modeli güçlü yükseliş öngörüyor"
            },
        ],
        "sell_signals": [],
        "hold_signals": [],
        "portfolio": {
            "total_equity": 100000.0,
            "daily_pnl": 1250.0,
            "daily_pnl_pct": 1.25,
            "holdings": [
                { "symbol": "THYAO", "quantity": 100, "avg_price": 275.0, "current_price": 285.5, "pnl": 1050.0, "pnl_pct": 3.8 },
            ]
        },
        "config": {
            "model_type": "multitask",
            "entry_threshold": 0.65
        }
    };
}
