document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const newsInput = document.getElementById('news-input');
    const resultArea = document.getElementById('result-area');
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');
    const verdict = document.getElementById('verdict');
    const stressPanel = document.getElementById('stress-panel');
    const reputationPanel = document.getElementById('reputation-panel');
    const stressResources = document.getElementById('stress-resources');
    const reputationContent = document.getElementById('reputation-content');

    analyzeBtn.addEventListener('click', async () => {
        const text = newsInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            displayResult(data);

            if (data.is_fake || data.stress_level === 'High') {
                loadStressResources();
            } else {
                stressPanel.classList.add('hidden');
            }

            loadReputationData();

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Credibility';
        }
    });

    function displayResult(data) {
        resultArea.classList.remove('hidden');

        const confidencePercent = Math.round(data.confidence * 100);
        let color = '#22c55e'; // Green
        let text = 'Likely Real';

        if (data.is_fake) {
            color = '#ef4444'; // Red
            text = 'Likely Fake';
        }

        // Show warning for short inputs
        if (data.warning) {
            color = '#f59e0b'; // Orange/amber
            text = '⚠️ Insufficient Data';
            verdict.innerHTML = `<div style="color: ${color}; margin-bottom: 0.5rem;">${text}</div>
                <div style="font-size: 0.9rem; font-weight: normal; color: #64748b;">
                    ${data.warning}
                    <br><br>
                    <strong>Tip:</strong> ${data.suggestion}
                </div>`;
            gaugeFill.style.width = `${confidencePercent}%`;
            gaugeFill.style.backgroundColor = color;
            gaugeValue.textContent = 'Need More Context';
        } else {
            gaugeFill.style.width = `${confidencePercent}%`;
            gaugeFill.style.backgroundColor = color;
            gaugeValue.textContent = `${confidencePercent}% Confidence`;
            verdict.textContent = text;
            verdict.style.color = color;
        }

        // Display Fact Checks
        const factCheckSection = document.getElementById('fact-check-section');
        const factCheckList = document.getElementById('fact-check-list');

        if (data.fact_checks && data.fact_checks.length > 0) {
            factCheckList.innerHTML = data.fact_checks.map(item => `
                <li>
                    <a href="${item.link}" target="_blank"><strong>${item.source}</strong>: ${item.title}</a>
                    <p>${item.snippet}</p>
                </li>
            `).join('');
            factCheckSection.classList.remove('hidden');
        } else {
            factCheckSection.classList.add('hidden');
        }

        // Advanced Analysis
        const advancedSection = document.getElementById('advanced-analysis-section');
        const sentimentLabel = document.getElementById('sentiment-label');
        const triggerWords = document.getElementById('trigger-words');

        if (data.sentiment_label) {
            sentimentLabel.textContent = `${data.sentiment_label} (Subjectivity: ${Math.round(data.subjectivity_score * 100)}%)`;

            if (data.triggers && data.triggers.length > 0) {
                triggerWords.innerHTML = data.triggers.map(t => `<span class="tag fake">${t}</span>`).join(' ');
            } else {
                triggerWords.textContent = "None detected";
            }
            advancedSection.classList.remove('hidden');
        }

        // NEW: Display Stress Analysis
        displayStressAnalysis(data.stress_analysis);

        // NEW: Display Extracted Entities
        displayEntities(data.entities);
    }

    function displayStressAnalysis(stressData) {
        if (!stressData) return;

        const stressSection = document.getElementById('stress-analysis-section');
        const stressScoreEl = document.getElementById('stress-score');
        const stressLevelEl = document.getElementById('stress-level');
        const riskFactorsEl = document.getElementById('risk-factors');
        const stressRecommendationsEl = document.getElementById('stress-recommendations');

        // Display stress score with color
        stressScoreEl.textContent = `${stressData.stress_score}/100`;
        stressScoreEl.style.color = stressData.stress_color;
        stressLevelEl.textContent = stressData.stress_level;
        stressLevelEl.style.color = stressData.stress_color;

        // Display risk factors as bars
        if (stressData.risk_factors) {
            const factors = stressData.risk_factors;
            riskFactorsEl.innerHTML = Object.entries(factors)
                .filter(([key, value]) => value > 0)
                .map(([key, value]) => `
                    <div class="risk-factor">
                        <span class="factor-name">${key}:</span>
                        <div class="factor-bar">
                            <div class="factor-fill" style="width: ${value * 100}%; background-color: ${getFactorColor(value)}"></div>
                        </div>
                        <span class="factor-value">${Math.round(value * 100)}%</span>
                    </div>
                `).join('');
        }

        // Display recommendations
        if (stressData.recommendations && stressData.recommendations.length > 0) {
            stressRecommendationsEl.innerHTML = stressData.recommendations.map(rec => `
                <div class="recommendation ${rec.priority.toLowerCase()}">
                    <span class="rec-icon">${rec.icon}</span>
                    <div class="rec-content">
                        <strong>${rec.text}</strong>
                        <p>${rec.action}</p>
                    </div>
                </div>
            `).join('');
        }

        stressSection.classList.remove('hidden');
    }

    function getFactorColor(value) {
        if (value >= 0.7) return '#dc2626';  // Red
        if (value >= 0.5) return '#f59e0b';  // Orange
        if (value >= 0.3) return '#fbbf24';  // Yellow
        return '#22c55e';  // Green
    }

    function displayEntities(entitiesData) {
        if (!entitiesData || entitiesData.total_count === 0) return;

        const entitiesSection = document.getElementById('entities-section');
        const entitiesList = document.getElementById('entities-list');
        const entitiesCount = document.getElementById('entities-count');

        entitiesCount.textContent = `${entitiesData.total_count} entities detected`;

        // Display primary subjects prominently
        if (entitiesData.primary_subjects && entitiesData.primary_subjects.length > 0) {
            const primaryHTML = entitiesData.primary_subjects.map(entity => `
                <div class="entity-card primary">
                    <span class="entity-icon">${entity.icon}</span>
                    <div class="entity-info">
                        <strong>${entity.name}</strong>
                        <span class="entity-type">${entity.type}</span>
                    </div>
                </div>
            `).join('');

            entitiesList.innerHTML = `
                <div class="primary-entities">
                    <h4>Main Subjects</h4>
                    <div class="entities-grid">${primaryHTML}</div>
                </div>
            `;
        }

        // Display all entities by category
        if (entitiesData.entities_by_category) {
            const categoryHTML = Object.entries(entitiesData.entities_by_category)
                .map(([category, entities]) => `
                    <div class="entity-category">
                        <strong>${category}:</strong>
                        ${entities.map(e => `<span class="entity-tag">${e}</span>`).join(' ')}
                    </div>
                `).join('');

            entitiesList.innerHTML += `
                <div class="all-entities">
                    <h4>All Detected Entities</h4>
                    ${categoryHTML}
                </div>
            `;
        }

        entitiesSection.classList.remove('hidden');
    }

    async function loadStressResources() {
        try {
            const response = await fetch('/api/stress-relief');
            const data = await response.json();

            if (data.resources && data.resources.length > 0) {
                stressResources.innerHTML = data.resources.map(r => {
                    let content = `<li class="stress-resource ${r.type || ''}">`;
                    content += `<strong>${r.title}</strong>`;
                    if (r.contact) {
                        content += `<div class="resource-contact">📞 ${r.contact}</div>`;
                    }
                    if (r.link) {
                        content += `<div><a href="${r.link}" target="_blank">Access Resource →</a></div>`;
                    }
                    if (r.description) {
                        content += `<p>${r.description}</p>`;
                    }
                    if (r.duration) {
                        content += `<span class="duration">⏱️ ${r.duration}</span>`;
                    }
                    if (r.priority === 'URGENT') {
                        content += `<span class="urgent-badge">🚨 URGENT</span>`;
                    }
                    content += `</li>`;
                    return content;
                }).join('');

                // Show stress level if available
                if (data.stress_level) {
                    const levelBadge = document.createElement('div');
                    levelBadge.className = `stress-level-badge ${data.stress_level.toLowerCase()}`;
                    levelBadge.textContent = `Stress Level: ${data.stress_level} (${data.stress_score}/100)`;
                    stressPanel.insertBefore(levelBadge, stressResources);
                }

                stressPanel.classList.remove('hidden');
            } else {
                stressPanel.classList.add('hidden');
            }
        } catch (e) {
            console.error("Failed to load stress resources", e);
        }
    }

    async function loadReputationData() {
        try {
            const response = await fetch('/api/reputation');
            const data = await response.json();

            reputationContent.innerHTML = `
                <p><strong>Entity:</strong> ${data.entity}</p>
                <p><strong>Reputation Score:</strong> ${data.score}/100</p>
                <ul>
                    ${data.action_items.map(item => `<li>${item}</li>`).join('')}
                </ul>
            `;
        } catch (e) {
            console.error("Failed to load reputation data", e);
        }
    }
    const navHome = document.getElementById('nav-home');
    const navDashboard = document.getElementById('nav-dashboard');
    const viewAnalyzer = document.getElementById('view-analyzer');
    const viewDashboard = document.getElementById('view-dashboard');
    const historyList = document.getElementById('history-list');
    let chartInstance = null;

    // Navigation
    navHome.addEventListener('click', () => {
        navHome.classList.add('active');
        navDashboard.classList.remove('active');
        viewAnalyzer.classList.remove('hidden');
        viewDashboard.classList.add('hidden');
    });

    navDashboard.addEventListener('click', () => {
        navDashboard.classList.add('active');
        navHome.classList.remove('active');
        viewDashboard.classList.remove('hidden');
        viewAnalyzer.classList.add('hidden');
        loadDashboardData();
    });

    async function loadDashboardData() {
        try {
            const response = await fetch('/api/history');
            const history = await response.json();

            renderHistoryList(history);
            renderChart(history);
        } catch (e) {
            console.error("Failed to load dashboard data", e);
        }
    }

    function renderHistoryList(history) {
        historyList.innerHTML = history.slice().reverse().map(item => `
            <li>
                <div>
                    <strong>${item.text_preview}</strong>
                    <br>
                    <small>${new Date(item.timestamp).toLocaleString()}</small>
                </div>
                <div>
                    <span class="tag ${item.is_fake ? 'fake' : 'real'}">
                        ${item.is_fake ? 'FAKE' : 'REAL'}
                    </span>
                    <br>
                    <small>${Math.round(item.confidence * 100)}% Conf.</small>
                </div>
            </li>
        `).join('');
    }

    function renderChart(history) {
        const ctx = document.getElementById('trendChart').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
        }

        const realCount = history.filter(h => !h.is_fake).length;
        const fakeCount = history.filter(h => h.is_fake).length;

        chartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Real News', 'Fake News'],
                datasets: [{
                    data: [realCount, fakeCount],
                    backgroundColor: ['#22c55e', '#ef4444'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                    }
                }
            }
        });
    }
});
