// trading_signal_app/app/static/js/main.js

document.addEventListener('DOMContentLoaded', () => {

    // --- Element Selectors ---
    const modelStatusBadge = document.getElementById('model-status-badge');
    const globalStatusAlert = document.getElementById('global-status-alert');
    const timeframeSelect = document.getElementById('timeframe-select');
    const categorySelect = document.getElementById('pair-category');
    const assetSelect = document.getElementById('asset-symbol');
    const generateSignalBtn = document.getElementById('generate-signal-btn');
    const singleSignalResult = document.getElementById('single-signal-result');
    const scanSpinner = document.getElementById('scan-spinner');
    const scanResultsBody = document.getElementById('scan-results-body');
    const scanButtons = document.querySelectorAll('.scan-btn');
    const allInteractiveElements = [timeframeSelect, categorySelect, assetSelect, generateSignalBtn, ...scanButtons];

    // --- Helper Functions ---

    /**
     * Displays a Bootstrap alert message.
     * @param {HTMLElement} container - The element to inject the alert into.
     * @param {string} message - The message to display.
     * @param {string} type - The Bootstrap alert type (e.g., 'danger', 'success').
     */
    const showAlert = (container, message, type = 'danger') => {
        container.innerHTML = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>`;
    };

    /**
     * Toggles a spinner on a button and disables it.
     * @param {HTMLButtonElement} button - The button element.
     * @param {boolean} show - True to show spinner, false to hide.
     */
    const toggleButtonSpinner = (button, show) => {
        const spinner = button.querySelector('.spinner-border');
        button.disabled = show;
        if (spinner) {
            spinner.classList.toggle('d-none', !show);
        }
    };

    /**
     * Formats the signal class for styling (e.g., 'signal-buy').
     * @param {string} signal - The signal string ('BUY', 'SELL', 'HOLD').
     * @returns {string} The CSS class name.
     */
    const getSignalClass = (signal) => {
        if (!signal) return '';
        return `signal-${signal.toLowerCase()}`;
    };

    // --- Core Application Logic ---

    /**
     * Checks the backend model status and enables/disables UI accordingly.
     */
    const checkModelStatus = async () => {
        try {
            const response = await fetch('/api/check_model_status');
            const data = await response.json();

            if (data.models_loaded) {
                modelStatusBadge.textContent = 'Loaded';
                modelStatusBadge.className = 'badge bg-success';
                allInteractiveElements.forEach(el => el.disabled = false);
                // Re-disable asset select until a category is chosen
                assetSelect.disabled = true;
                generateSignalBtn.disabled = true;
            } else {
                throw new Error("Models failed to load on the server.");
            }
        } catch (error) {
            modelStatusBadge.textContent = 'Failed';
            modelStatusBadge.className = 'badge bg-danger';
            showAlert(globalStatusAlert, `<strong>Critical Error:</strong> Could not connect to the model service. All features are disabled. Please check the server logs.`, 'danger');
            allInteractiveElements.forEach(el => el.disabled = true);
        }
    };

    /**
     * Populates the asset dropdown based on the selected category.
     */
    const handleCategoryChange = () => {
        const selectedCategory = categorySelect.value;
        assetSelect.innerHTML = '<option value="">Select Asset</option>'; // Reset
        assetSelect.disabled = true;
        generateSignalBtn.disabled = true;
        
        if (selectedCategory && assetClasses[selectedCategory]) {
            assetClasses[selectedCategory].forEach(symbol => {
                const option = new Option(symbol, symbol);
                assetSelect.add(option);
            });
            assetSelect.disabled = false;
        }
    };

    /**
     * Fetches and displays a signal for a single selected asset.
     */
    const handleGenerateSignal = async () => {
        const symbol = assetSelect.value;
        const timeframe = timeframeSelect.value;
        if (!symbol) return;

        toggleButtonSpinner(generateSignalBtn, true);
        singleSignalResult.innerHTML = ''; // Clear previous results

        try {
            const response = await fetch(`/api/generate_signal?symbol=${symbol}&timeframe=${timeframe}`);
            const data = await response.json();

            if (response.status !== 200) {
                 showAlert(singleSignalResult, `<strong>Error:</strong> ${data.error || 'Unknown error occurred.'}`);
                 return;
            }
            
            const signalClass = getSignalClass(data.signal);
            let resultHtml;

            if (data.signal === "HOLD") {
                 resultHtml = `
                    <div class="card bg-secondary-subtle text-center p-3">
                        <div class="card-body">
                             <h4 class="card-title ${signalClass}"><i class="fas fa-hand-paper me-2"></i>HOLD</h4>
                             <p class="mb-1"><strong>Symbol:</strong> ${data.symbol}</p>
                             <p class="mb-0"><strong>Confidence:</strong> ${data.confidence}</p>
                        </div>
                    </div>`;
            } else {
                 resultHtml = `
                    <div class="card border-${signalClass === 'signal-buy' ? 'success' : 'danger'}">
                        <div class="card-header bg-transparent d-flex justify-content-between align-items-center">
                            <h5 class="mb-0 ${signalClass}"><i class="fas fa-arrow-trend-${signalClass === 'signal-buy' ? 'up' : 'down'} me-2"></i>${data.signal} SIGNAL</h5>
                            <span class="badge bg-secondary">${data.symbol}</span>
                        </div>
                        <div class="card-body">
                            <ul class="list-group list-group-flush">
                                <li class="list-group-item d-flex justify-content-between"><strong>Confidence:</strong> <span>${data.confidence}</span></li>
                                <li class="list-group-item d-flex justify-content-between"><strong>Entry Price:</strong> <span>${data.entry_price}</span></li>
                                <li class="list-group-item d-flex justify-content-between"><strong>Take Profit:</strong> <span>${data.exit_price}</span></li>
                                <li class="list-group-item d-flex justify-content-between"><strong>Stop Loss:</strong> <span>${data.stop_loss} <small class="text-muted ms-2">${data.stop_loss_value}</small></span></li>
                                <li class="list-group-item d-flex justify-content-between"><strong>Timestamp:</strong> <span>${data.timestamp}</span></li>
                            </ul>
                        </div>
                    </div>`;
            }
            singleSignalResult.innerHTML = resultHtml;

        } catch (error) {
            showAlert(singleSignalResult, `<strong>Network Error:</strong> Could not fetch signal. The server might be down or busy.`);
        } finally {
            toggleButtonSpinner(generateSignalBtn, false);
        }
    };
    
    /**
     * Scans a market category and displays the results in a table.
     * @param {Event} e - The click event from the scan button.
     */
    const handleScanMarket = async (e) => {
        const button = e.target;
        const assetType = button.dataset.assetType;
        const timeframe = timeframeSelect.value;

        scanSpinner.classList.remove('d-none');
        scanResultsBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">Scanning... This can take up to 2 minutes. Please wait.</td></tr>`;
        
        try {
            const response = await fetch('/api/scan_market', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ asset_type: assetType, timeframe: timeframe })
            });

            // Check if the response is successful. If not, parse error message.
            if (!response.ok) { // .ok is true for HTTP statuses 200-299
                let errorMessage = `Server error: ${response.status} ${response.statusText}`;
                try {
                    // Try to parse a JSON error from the backend, which is the expected format
                    const errorData = await response.json();
                    errorMessage = errorData.error || JSON.stringify(errorData);
                } catch (jsonError) {
                    // If backend crashes (e.g., timeout), response might not be JSON.
                    // This is where "unexpected end of JSON input" happens. We catch it.
                    errorMessage = "The server took too long to respond and the request timed out. This can happen on free hosting. Please try again in a moment.";
                }
                throw new Error(errorMessage);
            }

            const results = await response.json(); // Now it's safe to parse JSON
            scanResultsBody.innerHTML = ''; // Clear "Scanning..." message

            if (results.length > 0) {
                results.forEach(signal => {
                    const signalClass = getSignalClass(signal.signal);
                    const row = `
                        <tr>
                            <td>${signal.symbol}</td>
                            <td class="${signalClass}"><i class="fas fa-circle me-2"></i>${signal.signal}</td>
                            <td>${signal.confidence}</td>
                            <td>${signal.entry_price}</td>
                            <td>${signal.stop_loss}</td>
                            <td>${signal.stop_loss_value}</td>
                            <td>${signal.timestamp}</td>
                        </tr>
                    `;
                    scanResultsBody.innerHTML += row;
                });
            } else {
                scanResultsBody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No strong BUY/SELL signals found in this category.</td></tr>`;
            }

        } catch (error) {
            const row = `<tr><td colspan="7" class="text-center text-danger">Error scanning market: ${error.message}</td></tr>`;
            scanResultsBody.innerHTML = row;
        } finally {
            scanSpinner.classList.add('d-none');
        }
    };

    // --- Event Listeners ---
    categorySelect.addEventListener('change', handleCategoryChange);
    assetSelect.addEventListener('change', () => {
        generateSignalBtn.disabled = !assetSelect.value;
    });
    generateSignalBtn.addEventListener('click', handleGenerateSignal);
    scanButtons.forEach(button => button.addEventListener('click', handleScanMarket));

    // --- Initial Load ---
    checkModelStatus();
});
