#!/usr/bin/env python3
"""Local browser interface for reviewing v2 trash-bin annotations.

The application intentionally never writes into ``data/merged_v2``.  It reads
images and YOLO labels only to display the selected object; manual state is
stored separately in ``review.csv`` and ``missing_objects.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / "reports" / "v2_review"
REVIEW_CSV = REVIEW_DIR / "review.csv"
BACKUP_CSV = REVIEW_DIR / "review_before_manual.csv"
MISSING_CSV = REVIEW_DIR / "missing_objects.csv"
IMAGES_ROOT = PROJECT_ROOT / "data" / "merged_v2" / "images"
LABELS_ROOT = PROJECT_ROOT / "data" / "merged_v2" / "labels"

REVIEW_FIELDS = [
    "split",
    "image",
    "object_number",
    "old_class",
    "converted_class",
    "suggested_class",
    "reason",
    "human_decision",
]
MISSING_FIELDS = ["split", "image", "comment", "resolved"]
SPLITS = {"train", "val", "test"}
DECISIONS = {"not_full", "full", "overflow", "unknown", "delete"}
CLASS_NAMES = ["not_full", "full", "overflow", "unknown"]
LOCK = threading.RLock()


def read_review() -> list[dict[str, str]]:
    with REVIEW_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REVIEW_FIELDS:
            raise ValueError(
                f"Unexpected review.csv columns: {reader.fieldnames!r}; expected {REVIEW_FIELDS!r}"
            )
        return [{field: row.get(field, "") for field in REVIEW_FIELDS} for row in reader]


def atomic_write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV durably, then atomically replace the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def update_decision(row_id: int, decision: str) -> dict[str, str]:
    if decision not in DECISIONS:
        raise ValueError("Unsupported decision")
    with LOCK:
        rows = read_review()
        if row_id < 0 or row_id >= len(rows):
            raise IndexError("Review row does not exist")
        # Preserve the untouched original exactly once, immediately before the
        # first manual mutation.  copy2 never overwrites an existing backup.
        if not BACKUP_CSV.exists():
            shutil.copy2(REVIEW_CSV, BACKUP_CSV)
        rows[row_id]["human_decision"] = decision
        atomic_write_csv(REVIEW_CSV, REVIEW_FIELDS, rows)
        return rows[row_id]


def read_missing() -> list[dict[str, str]]:
    """Read manual missed-object notes; an absent file means no notes yet."""
    if not MISSING_CSV.exists():
        return []
    with MISSING_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MISSING_FIELDS:
            raise ValueError("Unexpected missing_objects.csv columns")
        return [
            {field: item.get(field, "") for field in MISSING_FIELDS}
            for item in reader
        ]


def upsert_missing_object(row_id: int, comment: str) -> dict[str, str]:
    """Create or update one missed-object note per split/image.

    Old versions of the interface could append duplicates.  Updating an image
    therefore also collapses any existing duplicates for that exact image.
    """
    with LOCK:
        rows = read_review()
        if row_id < 0 or row_id >= len(rows):
            raise IndexError("Review row does not exist")
        row = rows[row_id]
        existing = read_missing()
        key = (row["split"], row["image"])
        matching = [item for item in existing if (item["split"], item["image"]) == key]
        note = {
            "split": row["split"], "image": row["image"],
            "comment": comment.strip(),
            # Keep a resolution flag set outside this application intact when
            # the reviewer only changes the explanatory comment.
            "resolved": matching[0]["resolved"] if matching else "",
        }
        existing = [item for item in existing if (item["split"], item["image"]) != key]
        existing.append(note)
        atomic_write_csv(MISSING_CSV, MISSING_FIELDS, existing)
        return note


def delete_missing_object(row_id: int) -> bool:
    """Remove the missed-object note for the image of this review row."""
    with LOCK:
        rows = read_review()
        if row_id < 0 or row_id >= len(rows):
            raise IndexError("Review row does not exist")
        row = rows[row_id]
        existing = read_missing()
        kept = [
            item for item in existing
            if (item["split"], item["image"]) != (row["split"], row["image"])
        ]
        removed = len(kept) != len(existing)
        if removed:
            atomic_write_csv(MISSING_CSV, MISSING_FIELDS, kept)
        return removed


def clean_part(value: str, allowed: set[str] | None = None) -> str:
    """Accept only simple filenames and known split values in file routes."""
    if not value or Path(value).name != value:
        raise ValueError("Invalid path component")
    if allowed is not None and value not in allowed:
        raise ValueError("Invalid value")
    return value


def get_bbox(row: dict[str, str]) -> dict[str, Any]:
    split = clean_part(row["split"], SPLITS)
    image = clean_part(row["image"])
    label = LABELS_ROOT / split / f"{Path(image).stem}.txt"
    try:
        object_number = int(row["object_number"])
    except ValueError as exc:
        raise ValueError("Invalid object number in review CSV") from exc
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if object_number < 1 or object_number > len(lines):
        raise ValueError(f"Object {object_number} is absent from {label.name}")
    parts = lines[object_number - 1].split()
    if len(parts) != 5:
        raise ValueError("Invalid YOLO label format")
    class_id = int(parts[0])
    coords = [float(value) for value in parts[1:]]
    if class_id not in range(len(CLASS_NAMES)) or len(coords) != 4:
        raise ValueError("Invalid label values")
    return {"bbox": coords, "label_class": CLASS_NAMES[class_id]}


def filters_from_query(query: dict[str, list[str]]) -> tuple[str, str, bool]:
    split = query.get("split", ["all"])[0]
    suggested = query.get("suggested", ["all"])[0]
    unresolved = query.get("unresolved", ["1"])[0] not in {"0", "false", "False"}
    if split != "all" and split not in SPLITS:
        raise ValueError("Invalid split filter")
    if suggested != "all" and suggested not in DECISIONS - {"delete"}:
        raise ValueError("Invalid suggested-class filter")
    return split, suggested, unresolved


def state_payload(query: dict[str, list[str]]) -> dict[str, Any]:
    split_filter, suggested_filter, unresolved_only = filters_from_query(query)
    requested_id = query.get("id", [""])[0]
    with LOCK:
        rows = read_review()
        notes = read_missing()
    selected_ids = [
        idx for idx, row in enumerate(rows)
        if (split_filter == "all" or row["split"] == split_filter)
        and (suggested_filter == "all" or row["suggested_class"] == suggested_filter)
        and (not unresolved_only or not row["human_decision"])
    ]
    requested: int | None = None
    if requested_id:
        try:
            requested = int(requested_id)
        except ValueError as exc:
            raise ValueError("Invalid review row number") from exc
        if requested < 0 or requested >= len(rows):
            raise ValueError("Review row does not exist")
        # A direct jump/back action must still permit correcting a resolved row,
        # even if the "unresolved only" filter would otherwise hide it.
        if requested not in selected_ids:
            selected_ids.append(requested)
            selected_ids.sort()
    current_id = requested if requested is not None else (selected_ids[0] if selected_ids else None)
    decision_counts = {name: 0 for name in DECISIONS}
    for row in rows:
        if row["human_decision"] in decision_counts:
            decision_counts[row["human_decision"]] += 1
    payload: dict[str, Any] = {
        "total": len(rows),
        "remaining": sum(not row["human_decision"] for row in rows),
        "decision_counts": decision_counts,
        "ids": selected_ids,
        "filters": {
            "split": split_filter, "suggested": suggested_filter,
            "unresolved": unresolved_only,
        },
        "current": None,
    }
    if current_id is not None:
        row = dict(rows[current_id])
        row["id"] = current_id
        row["review_number"] = current_id + 1
        row.update(get_bbox(row))
        row["missing_notes"] = [
            note for note in notes
            if note["split"] == row["split"] and note["image"] == row["image"]
        ]
        row["image_url"] = "/image?" + urllib.parse.urlencode({
            "split": row["split"], "image": row["image"],
        })
        payload["current"] = row
        payload["position"] = selected_ids.index(current_id) + 1
    else:
        payload["position"] = 0
    return payload


INDEX_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка разметки v2</title><style>
:root{color-scheme:dark;--bg:#10151c;--card:#1a222d;--line:#354355;--txt:#edf3fa;--muted:#a8b6c7;--blue:#2676d2;--good:#229b68;--warn:#c9851e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:16px system-ui,sans-serif}header{padding:15px 24px;border-bottom:1px solid var(--line);display:flex;gap:20px;align-items:center;flex-wrap:wrap}h1{font-size:21px;margin:0}main{max-width:1800px;margin:auto;padding:18px 24px}.filters,.stats,.buttons{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.filters{margin-bottom:14px}label{color:var(--muted)}select,input,textarea,button,a.download{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 10px;background:#111923;color:var(--txt)}button{cursor:pointer}.download{color:var(--txt);text-decoration:none}.layout{display:grid;grid-template-columns:minmax(420px,2fr) minmax(340px,1fr);gap:16px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}canvas{display:block;width:100%;background:#080b0e;border-radius:7px;max-height:64vh;object-fit:contain}.crop canvas{max-height:43vh}.meta{display:grid;grid-template-columns:max-content 1fr;gap:8px 13px;margin:0}.meta dt{color:var(--muted)}.meta dd{margin:0;overflow-wrap:anywhere}.buttons{margin-top:18px}.buttons button{font-size:18px;font-weight:700;min-width:142px;min-height:58px}.buttons button[data-decision="not_full"]{background:#19714d}.buttons button[data-decision="full"]{background:#2263a5}.buttons button[data-decision="overflow"]{background:#a8501b}.buttons button[data-decision="unknown"]{background:#6c4d8d}.buttons button[data-decision="delete"]{background:#9a3036}.secondary{background:#293545}.rules{line-height:1.43;color:#d9e2ed}.rules b{color:#fff}.hint,.muted{color:var(--muted)}.status{min-height:1.4em;color:#a6d8bd}.status.success{font-size:18px;font-weight:750;color:#65e7a7;background:#144832;border:1px solid #2a9563;border-radius:8px;padding:9px 12px}.hidden{display:none}.missing-notes{margin-top:12px;padding:10px 12px;background:#152b3c;border-left:4px solid #40b7ef;border-radius:6px}.missing-notes h3{font-size:16px;margin:0 0 7px}.missing-notes ul{margin:0;padding-left:22px;white-space:pre-wrap}.missing-notes .none{color:var(--muted)}dialog{width:min(620px,calc(100% - 28px));background:var(--card);color:var(--txt);border:1px solid var(--line);border-radius:12px;padding:20px;box-shadow:0 16px 60px #000}dialog::backdrop{background:#000a}dialog h2{margin-top:0}dialog textarea{display:block;width:100%;min-height:145px;margin:12px 0;resize:vertical}dialog .dialog-buttons{display:flex;gap:10px;justify-content:flex-end;flex-wrap:wrap}dialog .danger{background:#9a3036}@media(max-width:960px){.layout{grid-template-columns:1fr}main{padding:13px}.buttons button{flex:1;min-width:120px}}
</style></head><body tabindex="-1"><header><h1>Ручная проверка контейнеров v2</h1><strong id="progress">Загрузка…</strong><span id="remaining"></span><a class="download" href="/download/review.csv">Скачать review.csv</a></header>
<main><div class="filters card"><label><input id="unresolved" type="checkbox" checked> Только нерешённые</label><label>Split <select id="split"><option value="all">все</option><option>train</option><option>val</option><option>test</option></select></label><label>Предложенный класс <select id="suggested"><option value="all">все</option><option>not_full</option><option>full</option><option>overflow</option><option>unknown</option></select></label><label>Перейти к № <input id="goto" type="number" min="1" step="1" placeholder="1…1170" size="8"></label><button id="gotoBtn">Перейти</button><span class="hint">Горячие клавиши: 1–5, S, B, M</span></div>
<div class="stats card" id="stats"></div><p class="status" id="status"></p><section id="empty" class="card hidden">По выбранному фильтру объектов нет.</section>
<section id="content" class="layout"><div class="card"><h2>Исходный кадр</h2><canvas id="fullCanvas"></canvas><p class="hint">Зелёная рамка — проверяемый объект. Осматривайте также весь кадр на пропущенные контейнеры.</p><section class="missing-notes"><h3>Пропущенные объекты на этом изображении:</h3><ul id="missingNotes"><li class="none">Отметок пока нет.</li></ul></section></div><div><div class="card crop"><h2>Увеличенный фрагмент</h2><canvas id="cropCanvas"></canvas></div><div class="card"><h2>Объект</h2><dl class="meta" id="meta"></dl></div></div></section>
<section class="card"><h2>Решение</h2><div class="buttons"><button data-decision="not_full">1 · not_full</button><button data-decision="full">2 · full</button><button data-decision="overflow">3 · overflow</button><button data-decision="unknown">4 · unknown</button><button data-decision="delete">5 · delete</button><button class="secondary" id="skip">S · пропустить</button><button class="secondary" id="back">B · назад</button><button class="secondary" id="missing">M · пропущен контейнер</button></div><p class="hint">Можно назначить решение повторно, чтобы исправить принятое. «Пропущен контейнер» не меняет класс текущего объекта.</p></section>
<section class="card rules"><h2>Правила классов</h2><p><b>not_full</b> — содержимое видно, мусор находится ниже верхнего края.</p><p><b>full</b> — мусор дошёл до верхнего края, но не выступает выше него.</p><p><b>overflow</b> — мусор внутри контейнера выступает выше его верхнего края.</p><p><b>unknown</b> — содержимое невозможно оценить из-за закрытой крышки, перекрытия, темноты, размытия или слишком большого расстояния.</p><p><b>delete</b> — рамка не относится к мусорному контейнеру.</p><p>Мусор рядом с контейнером не влияет на класс и не размечается.</p></section></main>
<dialog id="missingDialog"><form method="dialog"><h2>Пропущенный контейнер</h2><p class="hint">Опишите видимый контейнер, который не получил рамку. Класс текущего объекта не изменится.</p><textarea id="missingComment" aria-label="Комментарий о пропущенном объекте" placeholder="Например: внизу кадра виден неразмеченный контейнер…"></textarea><div class="dialog-buttons"><button type="button" class="danger hidden" id="deleteMissing">Удалить отметку</button><button type="button" id="cancelMissing">Отмена</button><button type="button" id="saveMissing">Сохранить отметку</button></div></form></dialog>
<script>
const $=id=>document.getElementById(id); let state=null, activeId=null, history=[]; const full=$('fullCanvas'), crop=$('cropCanvas'), missingDialog=$('missingDialog'), missingComment=$('missingComment');
function query(id){const p=new URLSearchParams({split:$('split').value,suggested:$('suggested').value,unresolved:$('unresolved').checked?'1':'0'});if(id!==null&&id!==undefined)p.set('id',id);return '/api/state?'+p}
async function load(id=null, remember=false){$('status').textContent='';try{const r=await fetch(query(id));if(!r.ok)throw Error((await r.json()).error||'Ошибка загрузки');state=await r.json();const item=state.current;if(remember&&item&&activeId!==item.id)history.push(activeId);activeId=item?item.id:null;render();}catch(e){$('status').textContent=e.message}}
function render(){const item=state.current;$('progress').textContent=item?`Объект ${state.position} из ${state.ids.length} (запись ${item.review_number} из ${state.total})`:`0 из ${state.ids.length}`;$('remaining').textContent=`Осталось: ${state.remaining}`;$('stats').innerHTML=Object.entries(state.decision_counts).map(([k,v])=>`<span><b>${k}</b>: ${v}</span>`).join(' · ')+' · <span>без решения: <b>'+state.remaining+'</b></span>';$('content').classList.toggle('hidden',!item);$('empty').classList.toggle('hidden',!!item);if(!item)return;const info=[['Изображение',item.image],['Split',item.split],['Номер объекта',item.object_number],['Старый класс',item.old_class],['Преобразованный',item.converted_class],['Предложенный',item.suggested_class],['Текущий YOLO-класс',item.label_class],['Причина',item.reason],['Решение человека',item.human_decision||'—']];$('meta').innerHTML=info.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('');const notes=item.missing_notes||[];$('missingNotes').innerHTML=notes.length?notes.map(note=>`<li>${escapeHtml(note.comment)||'Комментарий не указан.'}</li>`).join(''):'<li class="none">Отметок пока нет.</li>';const img=new Image();img.onload=()=>draw(img,item.bbox);img.onerror=()=>{$('status').textContent='Не удалось открыть изображение';};img.src=item.image_url+'&_='+Date.now()}
function escapeHtml(value){const node=document.createElement('span');node.textContent=value;return node.innerHTML}
function setup(c,w,h){const d=Math.min(2,window.devicePixelRatio||1);c.width=Math.max(1,Math.round(w*d));c.height=Math.max(1,Math.round(h*d));c.style.aspectRatio=`${w}/${h}`;const x=c.getContext('2d');x.setTransform(d,0,0,d,0,0);return x}
function box(ctx,x,y,w,h){ctx.strokeStyle='#39e68a';ctx.lineWidth=Math.max(2,Math.min(w,h)/220);ctx.strokeRect(x,y,w,h);ctx.fillStyle='rgba(0,0,0,.72)';ctx.fillRect(x,y-22,125,21);ctx.fillStyle='#fff';ctx.font='15px system-ui';ctx.fillText('Текущий объект',x+4,y-6)}
function draw(img,b){const maxW=1100,maxH=760,scale=Math.min(1,maxW/img.naturalWidth,maxH/img.naturalHeight),w=Math.round(img.naturalWidth*scale),h=Math.round(img.naturalHeight*scale);let ctx=setup(full,w,h);ctx.drawImage(img,0,0,w,h);box(ctx,(b[0]-b[2]/2)*w,(b[1]-b[3]/2)*h,b[2]*w,b[3]*h);let x=(b[0]-b[2]/2)*img.naturalWidth,y=(b[1]-b[3]/2)*img.naturalHeight,bw=b[2]*img.naturalWidth,bh=b[3]*img.naturalHeight,pad=Math.max(35,Math.max(bw,bh)*.28);let sx=Math.max(0,x-pad),sy=Math.max(0,y-pad),ex=Math.min(img.naturalWidth,x+bw+pad),ey=Math.min(img.naturalHeight,y+bh+pad);const cs=Math.min(4,700/(ex-sx),580/(ey-sy));const cw=Math.round((ex-sx)*cs),ch=Math.round((ey-sy)*cs);ctx=setup(crop,cw,ch);ctx.drawImage(img,sx,sy,ex-sx,ey-sy,0,0,cw,ch);box(ctx,(x-sx)*cs,(y-sy)*cs,bw*cs,bh*cs)}
function next(){if(!state.ids.length)return;const i=state.ids.indexOf(activeId);load(state.ids[Math.min(i+1,state.ids.length-1)],true)}
async function decide(decision){if(activeId===null)return;const oldIndex=state.ids.indexOf(activeId);try{const r=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:activeId,decision})});if(!r.ok)throw Error((await r.json()).error||'Не удалось сохранить');$('status').textContent=`Сохранено: ${decision}`;await load(null,false);if(state.ids.length){await load(state.ids[Math.min(Math.max(oldIndex,0),state.ids.length-1)],false)}}catch(e){$('status').textContent=e.message}}
function currentMissingNote(){return state&&state.current&&(state.current.missing_notes||[])[0]}
function closeMissingDialog(){if(missingDialog.open)missingDialog.close();document.body.focus()}
function missing(){if(activeId===null)return;const note=currentMissingNote();missingComment.value=note?note.comment:'';$('deleteMissing').classList.toggle('hidden',!note);missingDialog.showModal();setTimeout(()=>missingComment.focus(),0)}
async function saveMissing(){if(activeId===null)return;try{const r=await fetch('/api/missing',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:activeId,comment:missingComment.value})});if(!r.ok)throw Error((await r.json()).error||'Не удалось сохранить отметку');closeMissingDialog();await load(activeId,false);$('status').textContent='Пропущенный объект записан';$('status').classList.add('success')}catch(e){$('status').textContent=e.message;$('status').classList.remove('success')}}
async function deleteMissing(){if(activeId===null)return;try{const r=await fetch('/api/missing/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:activeId})});if(!r.ok)throw Error((await r.json()).error||'Не удалось удалить отметку');closeMissingDialog();await load(activeId,false);$('status').textContent='Отметка о пропущенном объекте удалена';$('status').classList.add('success')}catch(e){$('status').textContent=e.message;$('status').classList.remove('success')}}
document.querySelectorAll('[data-decision]').forEach(b=>b.onclick=()=>decide(b.dataset.decision));$('skip').onclick=next;$('back').onclick=()=>{const prev=history.pop();if(prev!==undefined&&prev!==null)load(prev,false)};$('missing').onclick=missing;$('saveMissing').onclick=saveMissing;$('deleteMissing').onclick=deleteMissing;$('cancelMissing').onclick=closeMissingDialog;missingDialog.addEventListener('close',()=>document.body.focus());$('gotoBtn').onclick=()=>{const n=Number($('goto').value);if(n>=1&&n<=state.total)load(n-1,true);else $('status').textContent='Введите номер от 1 до '+state.total};['split','suggested','unresolved'].forEach(id=>$(id).addEventListener('change',()=>{history=[];load()}));$('goto').addEventListener('keydown',e=>{if(e.key==='Enter')$('gotoBtn').click()});document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)||missingDialog.open)return;const action={Digit1:()=>decide('not_full'),Digit2:()=>decide('full'),Digit3:()=>decide('overflow'),Digit4:()=>decide('unknown'),Digit5:()=>decide('delete'),KeyS:next,KeyB:()=>$('back').click(),KeyM:missing}[e.code];if(action){e.preventDefault();action()}});load();
</script></body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ReviewV2/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(self, content: bytes, content_type: str, status: int = 200,
                   download_name: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data: dict[str, Any], status: int = 200) -> None:
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.send_json(state_payload(query))
            elif parsed.path == "/image":
                split = clean_part(query.get("split", [""])[0], SPLITS)
                image = clean_part(query.get("image", [""])[0])
                image_path = IMAGES_ROOT / split / image
                if not image_path.is_file():
                    self.send_error_json("Image not found", 404)
                    return
                content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
                self.send_bytes(image_path.read_bytes(), content_type)
            elif parsed.path == "/download/review.csv":
                self.send_bytes(REVIEW_CSV.read_bytes(), "text/csv; charset=utf-8",
                                download_name="review.csv")
            else:
                self.send_error_json("Not found", 404)
        except (OSError, ValueError, IndexError) as exc:
            self.send_error_json(str(exc), 400)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Invalid request body")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            row_id = int(body["id"])
            if parsed.path == "/api/decision":
                result = update_decision(row_id, str(body["decision"]))
                self.send_json({"ok": True, "row": result})
            elif parsed.path == "/api/missing":
                note = upsert_missing_object(row_id, str(body.get("comment", "")))
                self.send_json({"ok": True, "note": note})
            elif parsed.path == "/api/missing/delete":
                self.send_json({"ok": True, "removed": delete_missing_object(row_id)})
            else:
                self.send_error_json("Not found", 404)
        except (KeyError, TypeError, ValueError, IndexError, OSError, json.JSONDecodeError) as exc:
            self.send_error_json(str(exc), 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local v2 annotation-review web app")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="local port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser automatically")
    args = parser.parse_args()
    if not REVIEW_CSV.is_file():
        raise SystemExit(f"review.csv was not found: {REVIEW_CSV}")
    if not IMAGES_ROOT.is_dir() or not LABELS_ROOT.is_dir():
        raise SystemExit("data/merged_v2/images or data/merged_v2/labels was not found")
    # Fail early if the review file has been edited into an incompatible shape.
    read_review()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Review v2 is running at {url}")
    print("Press Ctrl+C to stop the local server.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview v2 server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
