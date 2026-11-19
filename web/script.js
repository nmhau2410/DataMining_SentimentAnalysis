async function testSentiment() {
    const inputEl = document.getElementById('inputText');
    const text = inputEl.value.trim();
    if (!text) {
        alert('Vui lòng nhập câu phản hồi.');
        inputEl.focus();
        return;
    }
    const radioEl = document.querySelector('input[name="algorithm"]:checked');
    if (!radioEl) {
        alert('Vui lòng chọn một thuật toán.');
        return;
    }
    const selectedRadio = radioEl.value;
    const btn = document.querySelector('button');
    const originalBtnText = btn.innerText;
    btn.disabled = true;
    btn.innerText = 'Đang phân tích';
    const resultBox = document.getElementById('resultBox');
    resultBox.style.opacity = '0.4';
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                algorithm: selectedRadio
            })
        });
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || ('Lỗi máy chủ HTTP ' + response.status));
        }
        const data = await response.json();
        document.getElementById('resAlgo').innerText = data.algorithm;
        const resLabelEl = document.getElementById('resLabel');
        resLabelEl.innerText = data.label;
        resLabelEl.className = 'result-value ' + data.class_name;
        document.getElementById('resConf').innerText = data.confidence;
        document.getElementById('resProbs').innerText = data.probs;
    } catch (err) {
        alert('Không thể kết nối: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.innerText = originalBtnText;
        resultBox.style.opacity = '1';
    }
}
