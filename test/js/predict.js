// Initialize Lucide icons
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
});

function toggleTooltip(id, button) {
    // Hide all tooltips
    const tooltips = document.querySelectorAll('.tooltip');
    tooltips.forEach(tooltip => {
        tooltip.style.display = 'none';
    });

    // Get the selected tooltip
    const tooltip = document.getElementById(id);

    // Toggle the display of the selected tooltip
    if (tooltip.style.display === 'block') {
        tooltip.style.display = 'none'; // Hide if already visible
    } else {
        tooltip.style.display = 'block'; // Show if hidden
    }
}

function submitPrediction() {
    const pm25 = document.getElementById('pm25').value;
    const no2 = document.getElementById('no2').value;
    const o3 = document.getElementById('o3').value;
    const co = document.getElementById('co').value;

    fetch('http://127.0.0.1:8081/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            features: [pm25, no2, o3, co]
        }),
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('result');
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        } else {
            resultDiv.innerHTML = `<h3>Prediction: ${data.prediction}</h3><p>${data.suggestion}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    });
}