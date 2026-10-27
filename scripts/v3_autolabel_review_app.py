#!/usr/bin/env python3
"""Local-only reviewer for conservative V3 automatic candidates.

It binds to loopback by default and writes only the separate
``merged_v3_autolabeled`` snapshot.  Existing boxes are green and immutable in
this UI; proposed boxes are orange/purple and can be accepted, adjusted or
rejected one by one.
"""
from __future__ import annotations

import argparse, csv, json, mimetypes, os, tempfile, threading, urllib.parse
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged_v3_autolabeled"
SOURCE = ROOT / "data" / "merged_v3_work"
REPORT = ROOT / "reports" / "v3_autolabel"
QUEUE, CHANGES = REPORT / "manual_check_queue.csv", REPORT / "autolabel_changes.csv"
ACTIONS = REPORT / "review_actions.csv"
MANUAL = ROOT / "reports" / "v3_preparation" / "geometry_changes.csv"
NAMES = ("not_full", "full", "overflow", "unknown")
SPLITS = {"train", "val", "test"}
LOCK = threading.RLock()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as out:
        temp = Path(out.name); writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader(); writer.writerows(rows); out.flush(); os.fsync(out.fileno())
    try: os.replace(temp, path)
    finally: temp.unlink(missing_ok=True)


def safe(value: str, options: set[str] | None = None) -> str:
    if not value or Path(value).name != value or (options is not None and value not in options): raise ValueError("Недопустимый путь")
    return value


def paths(split: str, image: str) -> tuple[Path, Path]:
    split, image = safe(split, SPLITS), safe(image)
    file = DATA / "images" / split / image
    if not file.exists(): raise FileNotFoundError("Изображение отсутствует")
    return file, DATA / "labels" / split / f"{Path(image).stem}.txt"


def boxes(path: Path) -> list[dict[str, Any]]:
    result=[]
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        if not line.strip(): continue
        c,x,y,w,h=line.split(); result.append({"class_id":int(c),"x":float(x),"y":float(y),"width":float(w),"height":float(h)})
    return result


def valid(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict): raise ValueError("Некорректная рамка")
    c=int(item.get("class_id")); x,y,w,h=(float(item.get(key)) for key in ("x","y","width","height"))
    if c not in range(4) or not (0 < w <= 1 and 0 < h <= 1 and 0 <= x-w/2 and x+w/2 <= 1 and 0 <= y-h/2 and y+h/2 <= 1): raise ValueError("Рамка или класс вне допустимых границ")
    return {"class_id":c,"x":round(x,6),"y":round(y,6),"width":round(w,6),"height":round(h,6)}


def iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1,ay1,ax2,ay2=a['x']-a['width']/2,a['y']-a['height']/2,a['x']+a['width']/2,a['y']+a['height']/2
    bx1,by1,bx2,by2=b['x']-b['width']/2,b['y']-b['height']/2,b['x']+b['width']/2,b['y']+b['height']/2
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1)); union=a['width']*a['height']+b['width']*b['height']-inter
    return inter/union if union else 0


def manual_frames() -> set[tuple[str,str]]:
    return {(row['split'],row['image']) for row in read_csv(MANUAL)[1]}


def grouped() -> list[tuple[str,str]]:
    _, rows=read_csv(QUEUE); return list(dict.fromkeys((r['split'],r['image']) for r in rows if r['status']=='pending'))


def state(index: int) -> dict[str, Any]:
    groups=grouped()
    if not groups: return {"index":0,"total":0,"done":True,"classes":list(NAMES)}
    index=max(0,min(index,len(groups)-1)); split,image=groups[index]; im,label=paths(split,image); _,rows=read_csv(QUEUE)
    candidates=[]; existing=[]
    for row in rows:
        if (row['split'],row['image']) != (split,image) or row['status']!='pending': continue
        if row['change_type']=='add_box_candidate' and row['new_box']:
            candidates.append({"candidate_id":row['candidate_id'],"box":json.loads(row['new_box']),"class_id":int(row['class_id']),"class_name":row['class_name'],"level":row['level'],"source":row['source'],"confidence":row['confidence'],"method":row['method'],"basis":row['basis']})
        else: existing.append({"candidate_id":row['candidate_id'],"source":row['source'],"basis":row['basis']})
    return {"index":index,"total":len(groups),"split":split,"image":image,"image_url":f"/image?split={urllib.parse.quote(split)}&image={urllib.parse.quote(image)}","boxes":boxes(label),"candidates":candidates,"existing_tasks":existing,"classes":list(NAMES),"manual_protected":(split,image) in manual_frames()}


def atomic_label(path: Path, value: list[dict[str, Any]]) -> None:
    text=''.join(f"{b['class_id']} {b['x']:.6f} {b['y']:.6f} {b['width']:.6f} {b['height']:.6f}\n" for b in value)
    fd,tmp=tempfile.mkstemp(prefix='.label.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out: out.write(text);out.flush();os.fsync(out.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def commit(payload: dict[str,Any]) -> dict[str,Any]:
    split,image=safe(str(payload.get('split','')),SPLITS),safe(str(payload.get('image','')))
    im,label=paths(split,image)
    if (split,image) in manual_frames(): raise ValueError('Ручной эталонный кадр защищён: изменения запрещены')
    decisions=payload.get('decisions',{}); mark_existing=bool(payload.get('mark_existing_reviewed',False)); note=str(payload.get('note','')).strip()[:1000]
    if not isinstance(decisions,dict): raise ValueError('Некорректные решения')
    with LOCK:
        fields,rows=read_csv(QUEUE); current=boxes(label); actions=[]; added=0
        for row in rows:
            cid=row['candidate_id']
            if row['split']!=split or row['image']!=image or row['status']!='pending': continue
            decision=decisions.get(cid)
            if row['change_type']!='add_box_candidate':
                if mark_existing: row['status']='reviewed'; actions.append((cid,'existing_reviewed',''))
                continue
            if decision not in {'accept','correct','reject'}: continue
            if decision=='reject': row['status']='rejected'; actions.append((cid,'rejected','')); continue
            candidate=valid(payload.get('boxes',{}).get(cid, json.loads(row['new_box'])))
            if any(iou(candidate,b)>=.80 for b in current): raise ValueError(f'{cid}: рамка дублирует существующую (IoU ≥ 0.80)')
            current.append(candidate); row['status']='accepted'; added+=1; actions.append((cid,decision,json.dumps(candidate,ensure_ascii=False)))
        if actions:
            atomic_label(label,current); write_csv(QUEUE,fields,rows)
            # Keep the preparation ledger truthful after a person makes a
            # decision.  The original automatic proposal is never discarded;
            # its status and (when corrected) final rectangle are updated.
            change_fields, change_rows = read_csv(CHANGES)
            action_map = {cid: (action, box) for cid, action, box in actions}
            accepted_rows = []
            for change in change_rows:
                if change['change_id'] not in action_map:
                    continue
                action, final_box = action_map[change['change_id']]
                if action in {'accept', 'correct'}:
                    change['status'] = 'accepted_manual'; change['new_box'] = final_box
                    change['review_reason'] = 'Принято человеком в отдельном reviewer.'
                    accepted_rows.append(dict(change))
                elif action == 'rejected':
                    change['status'] = 'rejected_manual'; change['review_reason'] = 'Отклонено человеком в отдельном reviewer.'
            write_csv(CHANGES, change_fields, change_rows)
            if accepted_rows:
                added_file = REPORT / 'added_boxes.csv'; add_fields, add_rows = read_csv(added_file); add_rows.extend(accepted_rows); write_csv(added_file, add_fields, add_rows)
            action_fields=['timestamp','candidate_id','split','image','action','box','note']
            exists=ACTIONS.exists()
            with ACTIONS.open('a',encoding='utf-8',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=action_fields)
                if not exists: writer.writeheader()
                now=datetime.now(timezone.utc).isoformat()
                for cid,action,box in actions: writer.writerow({'timestamp':now,'candidate_id':cid,'split':split,'image':image,'action':action,'box':box,'note':note})
    return {'ok':True,'added':added,'remaining':len(grouped())}


PAGE=r'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Нейромусор — кандидаты v3</title><style>body{margin:0;background:#10151d;color:#edf2f7;font:15px system-ui,sans-serif}header{padding:14px 20px;background:#17212d;display:flex;gap:18px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0}main{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:14px;padding:14px}.stage{background:#070a0e;border-radius:10px;min-height:680px;display:grid;place-items:center;overflow:auto}canvas{max-width:100%;background:#111;touch-action:none}.side{background:#17212d;border-radius:10px;padding:14px}.meta,.candidate{white-space:pre-wrap;font-size:13px;color:#c8d5e5}.candidate{border:1px solid #3b5066;border-radius:7px;padding:8px;margin:8px 0;cursor:pointer}.candidate.sel{border-color:#ffe16b}.candidate.rejected{opacity:.45;text-decoration:line-through}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}button,select,textarea{font:inherit;border-radius:7px;border:1px solid #40556c;padding:9px;background:#233345;color:#fff}button{cursor:pointer}button.primary{background:#157f59;border:0}button.danger{background:#922f3c;border:0}textarea{width:100%;box-sizing:border-box;min-height:60px}.hint{font-size:12px;color:#b8c7d9;line-height:1.4}.status{min-height:20px;color:#91e5b9}@media(max-width:900px){main{grid-template-columns:1fr}.stage{min-height:400px}}</style><body><header><h1>V3 — проверка автоматических кандидатов</h1><b id="progress">Загрузка…</b><span id="status" class="status"></span></header><main><section class="stage"><canvas id="canvas" width="1100" height="700"></canvas></section><aside class="side"><div id="meta" class="meta"></div><div id="candidates"></div><div class="controls"><button id="accept">Принять</button><button id="reject" class="danger">Удалить</button><select id="class"></select><button id="existing">Подтвердить исходную задачу</button></div><label>Комментарий<textarea id="note" placeholder="Например: скорректировал до полного видимого корпуса"></textarea></label><div class="controls"><button id="prev">Назад</button><button id="next">Далее</button><button id="save" class="primary">Сохранить решения</button></div><p class="hint">Зелёные рамки — исходные, они здесь не редактируются. Оранжевые — medium, фиолетовые — low. Выберите кандидат, переместите его перетаскиванием или измените размер за маркер справа снизу; смена класса и движение означают «исправить». «Принять» добавит рамку в отдельную копию после сохранения, «Удалить» отклонит её.</p></aside></main><script>const q=s=>document.querySelector(s),c=q('#canvas'),ctx=c.getContext('2d');let st,img=new Image(),sel=-1,drag=null,view,dec={},edited={};const colors={medium:'#ff9f1a',low:'#bf7cff'};function msg(t,e=false){q('#status').textContent=t;q('#status').style.color=e?'#ff8c8c':'#91e5b9'}function clamp(v,a,b){return Math.max(a,Math.min(b,v))}function load(i){fetch('/api/state?index='+i).then(r=>r.json()).then(d=>{if(d.error)throw Error(d.error);st=d;sel=-1;dec={};edited={};q('#progress').textContent=d.done?'Очередь завершена':`${d.index+1} / ${d.total}`;q('#meta').textContent=d.done?'Нет ожидающих кадров':`${d.split} / ${d.image}\nАвтоматических кандидатов: ${d.candidates.length}\nИсходных задач: ${d.existing_tasks.length}`;q('#class').innerHTML=d.classes.map((x,i)=>`<option value="${i}">${x}</option>`).join('');list();img.onload=draw;img.src=d.image_url+'&v='+Date.now()}).catch(e=>msg(e.message,true))}function list(){if(!st||st.done)return;q('#candidates').innerHTML=st.candidates.map((x,i)=>`<div class="candidate ${i===sel?'sel':''} ${dec[x.candidate_id]==='reject'?'rejected':''}" data-i="${i}">#${i+1}: ${x.source}\n${x.method}\nconfidence: ${x.confidence}; ${x.level}\n${x.basis}</div>`).join('');document.querySelectorAll('.candidate').forEach(e=>e.onclick=()=>{sel=+e.dataset.i;q('#class').value=(edited[st.candidates[sel].candidate_id]||st.candidates[sel]).class_id;list();draw()})}function draw(){if(!st||st.done||!img.width)return;let sc=Math.min(c.width/img.width,c.height/img.height);view={w:img.width*sc,h:img.height*sc,x:(c.width-img.width*sc)/2,y:(c.height-img.height*sc)/2};ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(img,view.x,view.y,view.w,view.h);for(let b of st.boxes)paint(b,'#42d37b',2);st.candidates.forEach((x,i)=>{if(dec[x.candidate_id]!=='reject')paint(edited[x.candidate_id]||x.box, i===sel?'#ffe16b':colors[x.level],i===sel?4:3)});function paint(b,color,width){let x=view.x+(b.x-b.width/2)*view.w,y=view.y+(b.y-b.height/2)*view.h,w=b.width*view.w,h=b.height*view.h;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect(x,y,w,h);if(sel>=0&&b===(edited[st.candidates[sel].candidate_id]||st.candidates[sel].box))ctx.fillRect(x+w-5,y+h-5,10,10)}}function point(e){let r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height}}function norm(p){return{x:clamp((p.x-view.x)/view.w,0,1),y:clamp((p.y-view.y)/view.h,0,1)}}function hit(p){for(let i=st.candidates.length-1;i>=0;i--){let b=edited[st.candidates[i].candidate_id]||st.candidates[i].box,x=view.x+(b.x-b.width/2)*view.w,y=view.y+(b.y-b.height/2)*view.h,w=b.width*view.w,h=b.height*view.h;if(p.x>=x&&p.x<=x+w&&p.y>=y&&p.y<=y+h)return i}return-1}c.onpointerdown=e=>{if(!st||st.done)return;let p=point(e),i=hit(p);if(i<0)return;sel=i;let x=st.candidates[i],b=edited[x.candidate_id]||{...x.box,class_id:x.class_id};edited[x.candidate_id]={...b};q('#class').value=b.class_id;let hx=view.x+(b.x+b.width/2)*view.w,hy=view.y+(b.y+b.height/2)*view.h;drag={kind:Math.hypot(p.x-hx,p.y-hy)<18?'resize':'move',start:norm(p),box:{...b}};list();draw()};c.onpointermove=e=>{if(!drag||sel<0)return;let b=edited[st.candidates[sel].candidate_id],n=norm(point(e));if(drag.kind==='move'){b.x=clamp(drag.box.x+n.x-drag.start.x,b.width/2,1-b.width/2);b.y=clamp(drag.box.y+n.y-drag.start.y,b.height/2,1-b.height/2)}else{let cx=drag.box.x,cy=drag.box.y;b.x=(cx+n.x)/2;b.y=(cy+n.y)/2;b.width=Math.abs(n.x-cx);b.height=Math.abs(n.y-cy);b.x=clamp(b.x,b.width/2,1-b.width/2);b.y=clamp(b.y,b.height/2,1-b.height/2)}dec[st.candidates[sel].candidate_id]='correct';draw()};c.onpointerup=()=>drag=null;q('#class').onchange=()=>{if(sel<0)return;let x=st.candidates[sel],b=edited[x.candidate_id]||{...x.box};b.class_id=+q('#class').value;edited[x.candidate_id]=b;dec[x.candidate_id]='correct';draw()};q('#accept').onclick=()=>{if(sel<0)return;dec[st.candidates[sel].candidate_id]='accept';list();draw()};q('#reject').onclick=()=>{if(sel<0)return;dec[st.candidates[sel].candidate_id]='reject';list();draw()};q('#existing').onclick=()=>{st.existing_tasks.forEach(x=>dec[x.candidate_id]='existing');msg('Исходная задача будет отмечена проверенной после сохранения.')};q('#prev').onclick=()=>load(Math.max(0,st.index-1));q('#next').onclick=()=>load(Math.min(st.total-1,st.index+1));q('#save').onclick=()=>{let d={};Object.entries(dec).forEach(([k,v])=>d[k]=v==='existing'?undefined:v);fetch('/api/commit',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({split:st.split,image:st.image,decisions:d,boxes:edited,mark_existing_reviewed:Object.values(dec).includes('existing'),note:q('#note').value})}).then(r=>r.json()).then(x=>{if(x.error)throw Error(x.error);msg(`Сохранено; добавлено: ${x.added}. Осталось кадров: ${x.remaining}`);load(Math.min(st.index,x.remaining-1))}).catch(e=>msg(e.message,true))};load(0);</script>'''


class Handler(BaseHTTPRequestHandler):
    def json(self,payload:dict[str,Any],status=200)->None:
        raw=json.dumps(payload,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self)->None: # noqa: N802
        parsed=urllib.parse.urlparse(self.path); query=urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path=='/':
                raw=PAGE.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            elif parsed.path=='/api/state': self.json(state(int(query.get('index',['0'])[0])))
            elif parsed.path=='/image':
                image,_=paths(query.get('split',[''])[0],query.get('image',[''])[0]);self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(image.name)[0] or 'application/octet-stream');self.send_header('Content-Length',str(image.stat().st_size));self.end_headers();self.wfile.write(image.read_bytes())
            else:self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as error:self.json({'error':str(error)},400)
    def do_POST(self)->None: # noqa: N802
        if urllib.parse.urlparse(self.path).path!='/api/commit':self.send_error(HTTPStatus.NOT_FOUND);return
        try:size=int(self.headers.get('Content-Length','0'));self.json(commit(json.loads(self.rfile.read(size))))
        except Exception as error:self.json({'error':str(error)},400)
    def log_message(self,*_:object)->None:pass


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8094);args=parser.parse_args()
    for path in (DATA,QUEUE,CHANGES,MANUAL):
        if not path.exists():raise SystemExit(f'Required path is absent: {path}')
    server=ThreadingHTTPServer((args.host,args.port),Handler);print(f'V3 auto-label reviewer: http://{args.host}:{args.port}/');server.serve_forever()
if __name__=='__main__':main()
