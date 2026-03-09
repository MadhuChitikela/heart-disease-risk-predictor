document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultModal = document.getElementById('result-modal');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');
    
    // Result elements
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const resetBtn = document.getElementById('reset-btn');
    const metricsContainer = document.getElementById('metrics');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Indicate loading
        btnText.textContent = 'Processing...';
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;

        // Collect inputs in precise order: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        const age = document.getElementById('age').value;
        const bp = document.getElementById('trestbps').value;
        const chol = document.getElementById('chol').value;
        const maxHR = document.getElementById('thalach').value;

        const features = [
            parseFloat(age),
            parseFloat(document.getElementById('sex').value),
            parseFloat(document.getElementById('cp').value),
            parseFloat(bp),
            parseFloat(chol),
            parseFloat(document.getElementById('fbs').value),
            parseFloat(document.getElementById('restecg').value),
            parseFloat(maxHR),
            parseFloat(document.getElementById('exang').value),
            parseFloat(document.getElementById('oldpeak').value),
            parseFloat(document.getElementById('slope').value),
            parseFloat(document.getElementById('ca').value),
            parseFloat(document.getElementById('thal').value)
        ];

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ data: features })
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            showResult(data.prediction, { age, bp, chol, maxHR });
        } catch (error) {
            console.error('Prediction Error:', error);
            alert('Failed to connect to the prediction engine. Please make sure the API is running.');
        } finally {
            btnText.textContent = 'Run Prediction Engine';
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    function showResult(prediction, data) {
        form.classList.add('hidden');
        resultModal.classList.remove('hidden');

        // Build metrics to show what was evaluated
        metricsContainer.innerHTML = `
            <div class="metric-card"><span>Age</span><strong>${data.age}</strong></div>
            <div class="metric-card"><span>Resting BP</span><strong>${data.bp} mmHg</strong></div>
            <div class="metric-card"><span>Cholesterol</span><strong>${data.chol} mg/dl</strong></div>
            <div class="metric-card"><span>Max HR</span><strong>${data.maxHR} bpm</strong></div>
        `;

        if (prediction === 1) {
            resultIcon.innerHTML = '⚠️';
            resultTitle.textContent = 'Elevated Risk Detected';
            resultTitle.style.color = 'var(--danger)';
            resultDesc.textContent = "Based on the clinical markers provided, the XGBoost inference engine indicates a higher likelihood of cardiovascular complications. Immediate medical consultation and lifestyle adjustments are highly advised.";
        } else {
            resultIcon.innerHTML = '✅';
            resultTitle.textContent = 'Low Risk Profile';
            resultTitle.style.color = 'var(--success)';
            resultDesc.textContent = "The intelligent model predicts a low likelihood of heart disease based on current metrics. Continue maintaining a healthy lifestyle, diet, and regular checkups!";
        }
    }

    resetBtn.addEventListener('click', () => {
        form.reset();
        resultModal.classList.add('hidden');
        form.classList.remove('hidden');
    });
});
