function testSentiment() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) {
        alert('Vui lòng nhập câu phản hồi.');
        return;
    }
    const selectedRadio = document.querySelector('input[name="algorithm"]:checked').value;
    document.getElementById('resAlgo').innerText = selectedRadio;
    const lower = text.toLowerCase();
    let label = 'Tích cực (Positive)';
    let badgeClass = 'badge-pos';
    let conf = '96.5%';
    let probs = 'Tích cực: 96.5% | Trung tính: 2.3% | Tiêu cực: 1.2%';
    if (lower.includes('dở') || lower.includes('tệ') || lower.includes('chán') || lower.includes('khó hiểu') || lower.includes('sơ sài') || lower.includes('buồn ngủ')) {
        label = 'Tiêu cực (Negative)';
        badgeClass = 'badge-neg';
        conf = '94.8%';
        probs = 'Tích cực: 1.5% | Trung tính: 3.7% | Tiêu cực: 94.8%';
    } else if (lower.includes('tín chỉ') || lower.includes('lịch') || lower.includes('phòng') || lower.includes('thứ')) {
        label = 'Trung tính (Neutral)';
        badgeClass = 'badge-neu';
        conf = '88.0%';
        probs = 'Tích cực: 5.2% | Trung tính: 88.0% | Tiêu cực: 6.8%';
    }
    const resLabelEl = document.getElementById('resLabel');
    resLabelEl.innerText = label;
    resLabelEl.className = 'result-value ' + badgeClass;
    document.getElementById('resConf').innerText = conf;
    document.getElementById('resProbs').innerText = probs;
}
