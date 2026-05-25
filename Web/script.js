
var themeBtn = document.getElementById('themeToggle');
themeBtn.addEventListener('click', function () {
  if (document.body.classList.contains('light')) {
    document.body.classList.remove('light');
    themeBtn.textContent = '☀️';
  } else {
    document.body.classList.add('light');
    themeBtn.textContent = '🌙';
  }
});

var allTabs = document.querySelectorAll('.tab[data-tab], .mobile-tab[data-tab]');
var panels  = document.querySelectorAll('.panel');

allTabs.forEach(function (tab) {
  tab.addEventListener('click', function () {
    var target = tab.getAttribute('data-tab');
    if (!target) return;
    allTabs.forEach(function (t) {
      t.classList.toggle('active', t.getAttribute('data-tab') === target);
    });
    panels.forEach(function (p) {
      p.classList.toggle('active', p.id === 'panel-' + target);
    });
  });
});

var EMOTIONS = {
  'angry':             '😠',
  'disgust':           '🤢',
  'fear':              '😨',
  'happy':             '😊',
  'neutral':           '😐',
  'pleasant surprise': '😲',
  'sad':               '😢'
};

function renderResult(boxId, emotion, confidence, allProbs) {
  var box   = document.getElementById(boxId);
  var key   = emotion.toLowerCase();
  var emoji = EMOTIONS[key] || '🎭';
  var label = emotion.charAt(0).toUpperCase() + emotion.slice(1);
  var pct   = (confidence * 100).toFixed(2) + '%';
  var otherChips = '';
  if (allProbs) {
    var sorted = Object.entries(allProbs)
      .filter(function(e) { return e[0].toLowerCase() !== key; })
      .sort(function(a, b) { return b[1] - a[1]; });

    otherChips = '<div class="all-probs">';
    sorted.forEach(function(entry) {
      var eName  = entry[0];
      var eVal   = entry[1].toFixed(2) + '%';
      var eEmoji = EMOTIONS[eName.toLowerCase()] || '🎭';
      var eLabel = eName.charAt(0).toUpperCase() + eName.slice(1);
      otherChips +=
        '<div class="prob-chip">' +
          '<span class="prob-chip-emoji">' + eEmoji + '</span>' +
          '<span class="prob-chip-label">' + eLabel + '</span>' +
          '<span class="prob-chip-pct">' + eVal + '</span>' +
        '</div>';
    });
    otherChips += '</div>';
  }

  box.innerHTML =
    '<div class="result-filled">' +
      '<span class="result-emoji">' + emoji + '</span>' +
      '<div class="result-label">'  + label + '</div>'  +
      '<div class="result-pct">'    + pct   + '</div>'  +
      (otherChips ? '<div class="result-divider"></div>' + otherChips : '') +
    '</div>';
}

function renderError(boxId, msg) {
  document.getElementById(boxId).innerHTML =
    '<div class="result-empty">' +
      '<span class="result-empty-icon">⚠️</span>' +
      '<p>' + (msg || 'Error connecting to server.') + '</p>' +
    '</div>';
}

function setupFileUpload(dropId, inputId, fileInfoId, fileNameId, removeId) {
  var drop      = document.getElementById(dropId);
  var input     = document.getElementById(inputId);
  var fileInfo  = document.getElementById(fileInfoId);
  var fileName  = document.getElementById(fileNameId);
  var removeBtn = document.getElementById(removeId);

  drop.addEventListener('click', function (e) {
    if (e.target === input) return;
    input.click();
  });

  input.addEventListener('change', function () {
    if (input.files && input.files[0]) {
      fileName.textContent = input.files[0].name;
      fileInfo.classList.remove('hidden');
      drop.style.display = 'none';
    }
  });

  removeBtn.addEventListener('click', function () {
    input.value = '';
    fileName.textContent = 'No file selected';
    fileInfo.classList.add('hidden');
    drop.style.display = '';
  });
}

setupFileUpload('speechDrop', 'speechFile', 'speechFileInfo', 'speechFileName', 'speechRemove');
setupFileUpload('fusionDrop', 'fusionFile', 'fusionFileInfo', 'fusionFileName', 'fusionRemove');

document.getElementById('speechAnalyze').addEventListener('click', function () {
  var file = document.getElementById('speechFile').files[0];
  if (!file) { alert('Please select an audio file first.'); return; }

  var box = document.getElementById('speechResult');
  box.innerHTML = '<div class="result-empty"><span class="result-empty-icon">⏳</span><p>Analyzing...</p></div>';

  var fd = new FormData();
  fd.append('audio', file);

  fetch('http://127.0.0.1:5000/predict/speech', { method: 'POST', body: fd })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { renderError('speechResult', d.error); return; }
      renderResult('speechResult', d.emotion, d.confidence, d.all_probs);
    })
    .catch(function (err) { renderError('speechResult', 'Server error: ' + err.message); });
});

document.getElementById('textAnalyze').addEventListener('click', function () {
  var text = document.getElementById('textInput').value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  var box = document.getElementById('textResult');
  box.innerHTML = '<div class="result-empty"><span class="result-empty-icon">⏳</span><p>Analyzing...</p></div>';

  fetch('http://127.0.0.1:5000/predict/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: text })
  })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { renderError('textResult', d.error); return; }
      renderResult('textResult', d.emotion, d.confidence, d.all_probs);
    })
    .catch(function (err) { renderError('textResult', 'Server error: ' + err.message); });
});

document.getElementById('fusionAnalyze').addEventListener('click', function () {
  var file = document.getElementById('fusionFile').files[0];
  var text = document.getElementById('fusionText').value.trim();
  if (!file) { alert('Please select an audio file first.'); return; }
  if (!text) { alert('Please enter transcript text first.'); return; }

  var box = document.getElementById('fusionResult');
  box.innerHTML = '<div class="result-empty"><span class="result-empty-icon">⏳</span><p>Analyzing...</p></div>';

  var fd = new FormData();
  fd.append('audio', file);
  fd.append('text', text);

  fetch('http://127.0.0.1:5000/predict/fusion', { method: 'POST', body: fd })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { renderError('fusionResult', d.error); return; }
      renderResult('fusionResult', d.emotion, d.confidence, d.all_probs);
    })
    .catch(function (err) { renderError('fusionResult', 'Server error: ' + err.message); });
});
