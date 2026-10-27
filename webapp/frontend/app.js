const state = { file: null, models: [] };
const $ = (selector) => document.querySelector(selector);
const errorBox = $('#error'), statusBox = $('#status');

function message(text, type = 'status') { const node = type === 'error' ? errorBox : statusBox; node.textContent = text; node.hidden = !text; }
function setBusy(isBusy) { $('#predict').disabled = isBusy || !state.file || !$('#model').value; $('#predict').textContent = isBusy ? 'Распознавание…' : 'Распознать'; }
function setFile(file) {
  message('', 'error');
  if (!file) return;
  if (!['image/jpeg','image/png','image/webp'].includes(file.type) || file.size > 15 * 1024 * 1024) { message('Нужен JPG, PNG или WEBP размером не более 15 МБ.', 'error'); return; }
  state.file = file; $('#file-info').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} МБ`; $('#file-info').hidden = false; setBusy(false);
}
async function loadModels() {
  const response = await fetch('/api/models'); if (!response.ok) throw new Error('Список моделей недоступен');
  state.models = (await response.json()).models.filter((model) => model.status === 'available');
  $('#model').innerHTML = state.models.map((model) => `<option value="${model.id}">${model.name}</option>`).join(''); $('#model').disabled = !state.models.length; setBusy(false);
}
function renderResult(result) {
  $('#result').hidden = false; $('#original').src = `${result.original_image_url}?v=${result.result_id}`; $('#processed').src = `${result.processed_image_url}?v=${result.result_id}`; $('#download').href = result.download_url;
  $('#result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
$('#image').addEventListener('change', (event) => setFile(event.target.files[0]));
['dragenter','dragover'].forEach((eventName) => $('#drop-zone').addEventListener(eventName, (event) => { event.preventDefault(); $('#drop-zone').classList.add('drag'); }));
['dragleave','drop'].forEach((eventName) => $('#drop-zone').addEventListener(eventName, (event) => { event.preventDefault(); $('#drop-zone').classList.remove('drag'); }));
$('#drop-zone').addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));
for (const id of ['confidence','iou']) $('#' + id).addEventListener('input', (event) => { $('#' + id + '-value').textContent = Number(event.target.value).toFixed(2); });
$('#predict').addEventListener('click', async () => {
  if (!state.file) return; message(''); message('Модель обрабатывает изображение…'); setBusy(true);
  const body = new FormData(); body.append('model_id', $('#model').value); body.append('image', state.file); body.append('confidence', $('#confidence').value); body.append('iou', $('#iou').value);
  try { const response = await fetch('/api/predict', { method: 'POST', body }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Ошибка распознавания'); renderResult(payload); message(''); } catch (error) { message(error.message || 'Не удалось выполнить распознавание.', 'error'); message(''); } finally { setBusy(false); }
});
loadModels().catch((error) => { message(error.message, 'error'); $('#model').innerHTML = '<option>Модели недоступны</option>'; });
