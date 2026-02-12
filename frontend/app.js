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

        gaugeFill.style.width = `${confidencePercent}%`;
        gaugeFill.style.backgroundColor = color;
        gaugeValue.textContent = `${confidencePercent}% Confidence`;
        verdict.textContent = text;
        verdict.style.color = color;

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
    }

    async function loadStressResources() {
        try {
            const response = await fetch('/api/stress-relief');
            const resources = await response.json();

            stressResources.innerHTML = resources.map(r =>
                `<li><a href="${r.link}" target="_blank">${r.title}</a>: ${r.desc}</li>`
            ).join('');

            stressPanel.classList.remove('hidden');
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
