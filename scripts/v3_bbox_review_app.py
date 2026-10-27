#!/usr/bin/env python3
"""Local browser editor for v3 YOLO boxes; it writes only merged_v3_work."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged_v3_work"
REPORTS = ROOT / "reports" / "v3_preparation"
QUEUE = REPORTS / "manual_review_queue.csv"
HISTORY = REPORTS / "geometry_changes.csv"
ADDED = REPORTS / "missing_boxes_added.csv"
NAMES = ("not_full", "full", "overflow", "unknown")
SPLITS = {"train", "val", "test"}
LOCK = threading.RLock()


def atomic_text(path: Path, content: str) -> None:
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_component(value: str, allowed: set[str] | None = None) -> str:
    if not value or Path(value).name != value or (allowed is not None and value not in allowed):
        raise ValueError("Недопустимый путь")
    return value


def paths(split: str, image: str) -> tuple[Path, Path]:
    split, image = safe_component(split, SPLITS), safe_component(image)
    image_path = next((path for path in (DATA / "images" / split).glob(f"{Path(image).stem}.*") if path.name == image), None)
    if image_path is None:
        raise FileNotFoundError("Изображение отсутствует")
    return image_path, DATA / "labels" / split / f"{Path(image).stem}.txt"


def read_boxes(label: Path) -> list[dict[str, float | int]]:
    if not label.exists():
        return []
    boxes: list[dict[str, float | int]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Некорректная строка в {label.name}")
        class_id = int(parts[0]); x, y, width, height = map(float, parts[1:])
        boxes.append({"class_id": class_id, "x": x, "y": y, "width": width, "height": height})
    return boxes


def validate_boxes(value: Any) -> list[dict[str, float | int]]:
    if not isinstance(value, list):
        raise ValueError("Нужен список рамок")
    boxes = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Некорректная рамка")
        class_id = int(item.get("class_id"))
        x, y, width, height = (float(item.get(name)) for name in ("x", "y", "width", "height"))
        if class_id not in range(4) or not (0 < width <= 1 and 0 < height <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("Класс или координаты вне допустимого диапазона")
        if x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1:
            raise ValueError("Рамка должна целиком находиться в кадре")
        boxes.append({"class_id": class_id, "x": round(x, 6), "y": round(y, 6), "width": round(width, 6), "height": round(height, 6)})
    return boxes


def queue_state(index: int) -> dict[str, Any]:
    _, rows = read_csv(QUEUE)
    if not rows:
        raise ValueError("Очередь пуста")
    index = max(0, min(index, len(rows) - 1))
    current = dict(rows[index])
    image_path, label = paths(current["split"], current["image"])
    return {"index": index, "total": len(rows), "reviewed": sum(row["status"] == "reviewed" for row in rows), "item": current, "boxes": read_boxes(label), "image_url": f"/image?split={urllib.parse.quote(current['split'])}&image={urllib.parse.quote(current['image'])}", "image_name": image_path.name, "classes": list(NAMES)}


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(row)
        handle.flush(); os.fsync(handle.fileno())


def save(payload: dict[str, Any]) -> dict[str, Any]:
    split = safe_component(str(payload.get("split", "")), SPLITS)
    image = safe_component(str(payload.get("image", "")))
    note = str(payload.get("note", "")).strip()[:1000]
    boxes = validate_boxes(payload.get("boxes"))
    _, label = paths(split, image)
    with LOCK:
        before = read_boxes(label)
        serialised = "".join(f"{int(box['class_id'])} {box['x']:.6f} {box['y']:.6f} {box['width']:.6f} {box['height']:.6f}\n" for box in boxes)
        atomic_text(label, serialised)
        now = datetime.now(timezone.utc).isoformat()
        history_fields, queue_rows = read_csv(QUEUE)
        for row in queue_rows:
            if row["split"] == split and row["image"] == image:
                row["status"] = "reviewed"
        write_csv(QUEUE, history_fields, queue_rows)
        before_json, after_json = json.dumps(before, ensure_ascii=False), json.dumps(boxes, ensure_ascii=False)
        append_csv(HISTORY, ["timestamp", "split", "image", "action", "before", "after", "note"], {"timestamp": now, "split": split, "image": image, "action": "confirmed_full_frame_boxes", "before": before_json, "after": after_json, "note": note})
        if len(boxes) > len(before):
            for box in boxes[len(before):]:
                append_csv(ADDED, ["timestamp", "split", "image", "class", "x", "y", "width", "height", "note"], {"timestamp": now, "split": split, "image": image, "class": NAMES[int(box["class_id"])], "x": str(box["x"]), "y": str(box["y"]), "width": str(box["width"]), "height": str(box["height"]), "note": note})
    return {"ok": True, "boxes": boxes}


PAGE = r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Нейромусор — проверка v3</title><style>
body{margin:0;background:#10151d;color:#edf2f7;font:15px system-ui,sans-serif}header{padding:14px 20px;background:#17212d;display:flex;gap:18px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0}main{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px;padding:14px}.stage{background:#070a0e;border-radius:10px;min-height:680px;display:grid;place-items:center;overflow:auto}canvas{max-width:100%;height:auto;background:#111;touch-action:none}.side{background:#17212d;border-radius:10px;padding:14px}.meta{white-space:pre-wrap;color:#b8c7d9;font-size:13px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}button,select,input,textarea{font:inherit;border-radius:7px;border:1px solid #40556c;padding:9px;background:#233345;color:#fff}button{cursor:pointer}button.primary{background:#157f59;border:0}button.danger{background:#922f3c;border:0}textarea{width:100%;box-sizing:border-box;min-height:70px}.hint{font-size:12px;color:#b8c7d9;line-height:1.4}.status{min-height:22px;color:#91e5b9}.selected{color:#ffdb70}@media(max-width:900px){main{grid-template-columns:1fr}.stage{min-height:400px}}
</style><body><header><h1>V3 — ручная проверка рамок</h1><b id="progress">Загрузка…</b><span id="status"></span></header><main><section class="stage"><canvas id="canvas" width="1100" height="700"></canvas></section><aside class="side"><div id="meta" class="meta"></div><div class="controls"><button id="select">Выбор рамки</button><button id="add">Добавить рамку</button><button id="delete" class="danger">Удалить выбранную</button><select id="class"></select></div><label>Комментарий к подтверждению<textarea id="note" placeholder="Например: рамка расширена до корпуса контейнера"></textarea></label><div class="controls"><button id="prev">Назад</button><button id="next">Далее</button><button id="save" class="primary">Сохранить</button></div><p class="hint">Клик по рамке — выбор. Перетащите рамку за центр для перемещения, за жёлтый маркер справа снизу — для изменения размера. В режиме «Добавить» протяните новую рамку. Сохранение записывает label только в <code>merged_v3_work</code> и историю изменений.</p></aside></main><script>
const c=document.querySelector('#canvas'),ctx=c.getContext('2d'),q=s=>document.querySelector(s);let st=null,img=new Image(),boxes=[],sel=-1,mode='select',drag=null,view={x:0,y:0,w:1,h:1};const colors=['#43d17e','#47a6ff','#ff694f','#c184ff'];
function msg(t,err=false){q('#status').textContent=t;q('#status').style.color=err?'#ff8c8c':'#91e5b9'}function clamp(v,a,b){return Math.max(a,Math.min(b,v))}function load(i){fetch('/api/state?index='+i).then(r=>r.json()).then(d=>{if(d.error)throw Error(d.error);st=d;boxes=d.boxes.map(x=>({...x}));sel=-1;q('#progress').textContent=`${d.index+1} / ${d.total}; подтверждено ${d.reviewed}`;q('#meta').textContent=`${d.item.source}\n${d.item.split} / ${d.item.image}\nОбъект: ${d.item.object_index||'весь кадр'}\nПричина: ${d.item.reason}\nСтатус: ${d.item.status}`;q('#class').innerHTML=d.classes.map((x,i)=>`<option value="${i}">${x}</option>`).join('');img.onload=draw;img.src=d.image_url+'&v='+Date.now()}).catch(e=>msg(e.message,true))}
function draw(){if(!st||!img.width)return;const scale=Math.min(c.width/img.width,c.height/img.height);view={w:img.width*scale,h:img.height*scale,x:(c.width-img.width*scale)/2,y:(c.height-img.height*scale)/2};ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(img,view.x,view.y,view.w,view.h);boxes.forEach((b,i)=>{let x=view.x+(b.x-b.width/2)*view.w,y=view.y+(b.y-b.height/2)*view.h,w=b.width*view.w,h=b.height*view.h;ctx.lineWidth=i===sel?4:2;ctx.strokeStyle=i===sel?'#ffe16b':colors[b.class_id];ctx.strokeRect(x,y,w,h);ctx.fillStyle=ctx.strokeStyle;ctx.fillText(st.classes[b.class_id],x+3,Math.max(12,y-4));if(i===sel){ctx.fillRect(x+w-6,y+h-6,12,12)}})}
function point(e){let r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height}}function norm(p){return{x:clamp((p.x-view.x)/view.w,0,1),y:clamp((p.y-view.y)/view.h,0,1)}}function hit(p){for(let i=boxes.length-1;i>=0;i--){let b=boxes[i],x=view.x+(b.x-b.width/2)*view.w,y=view.y+(b.y-b.height/2)*view.h,w=b.width*view.w,h=b.height*view.h;if(p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h)return i}return -1}
c.onpointerdown=e=>{let p=point(e),n=norm(p);if(mode==='add'){boxes.push({class_id:+q('#class').value,x:n.x,y:n.y,width:.01,height:.01});sel=boxes.length-1;drag={kind:'add',start:n};draw();return}let i=hit(p);if(i<0){sel=-1;draw();return}sel=i;let b=boxes[i],hx=view.x+(b.x+b.width/2)*view.w,hy=view.y+(b.y+b.height/2)*view.h;drag={kind:Math.hypot(p.x-hx,p.y-hy)<18?'resize':'move',start:n,box:{...b}};q('#class').value=b.class_id;draw()};
c.onpointermove=e=>{if(!drag||sel<0)return;let n=norm(point(e)),b=boxes[sel];if(drag.kind==='move'){b.x=clamp(drag.box.x+n.x-drag.start.x,b.width/2,1-b.width/2);b.y=clamp(drag.box.y+n.y-drag.start.y,b.height/2,1-b.height/2)}else{let cx=drag.kind==='add'?drag.start.x:drag.box.x,cy=drag.kind==='add'?drag.start.y:drag.box.y;b.x=(cx+n.x)/2;b.y=(cy+n.y)/2;b.width=Math.abs(n.x-cx);b.height=Math.abs(n.y-cy);b.x=clamp(b.x,b.width/2,1-b.width/2);b.y=clamp(b.y,b.height/2,1-b.height/2)}draw()};c.onpointerup=()=>{drag=null};
q('#select').onclick=()=>{mode='select';msg('Режим выбора рамки')};q('#add').onclick=()=>{mode='add';msg('Протяните новую рамку по полному видимому корпусу')};q('#delete').onclick=()=>{if(sel>=0){boxes.splice(sel,1);sel=-1;draw()}};q('#class').onchange=()=>{if(sel>=0){boxes[sel].class_id=+q('#class').value;draw()}};q('#prev').onclick=()=>load(Math.max(0,st.index-1));q('#next').onclick=()=>load(Math.min(st.total-1,st.index+1));q('#save').onclick=()=>{if(!st)return;fetch('/api/save',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({split:st.item.split,image:st.item.image,boxes,note:q('#note').value})}).then(r=>r.json()).then(d=>{if(d.error)throw Error(d.error);msg('Сохранено. Все записи очереди для этого кадра отмечены как проверенные.');st.item.status='reviewed'}).catch(e=>msg(e.message,true))};load(0);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(content))); self.end_headers(); self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path); query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                body = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            elif parsed.path == "/api/state": self.send_json(queue_state(int(query.get("index", ["0"])[0])))
            elif parsed.path == "/image":
                image, _ = paths(query.get("split", [""])[0], query.get("image", [""])[0]); content_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"; self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(image.stat().st_size)); self.end_headers(); self.wfile.write(image.read_bytes())
            else: self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error: self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:  # noqa: N802
        if urllib.parse.urlparse(self.path).path != "/api/save": self.send_error(HTTPStatus.NOT_FOUND); return
        try:
            size = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(size)); self.send_json(save(payload))
        except Exception as error: self.send_json({"error": str(error)}, 400)

    def log_message(self, *_: object) -> None: pass


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8093); args = parser.parse_args()
    for required in (DATA, QUEUE, HISTORY, ADDED):
        if not required.exists(): raise SystemExit(f"Required v3 file is absent: {required}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"V3 reviewer: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__": main()
