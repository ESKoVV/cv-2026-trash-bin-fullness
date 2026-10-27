#!/usr/bin/env python3
"""Loopback-only reviewer for strict same-scene transfer groups.

No action is made on page load.  Applying a group records an exact before/after
snapshot and can be rolled back as long as no later edit changed that label.
"""
from __future__ import annotations
import argparse,csv,json,mimetypes,os,tempfile,threading,urllib.parse
from datetime import datetime,timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'merged_v3_autolabeled';PREP=ROOT/'reports'/'v3_preparation';REPORT=ROOT/'reports'/'v3_autolabel_v2';GROUPS=REPORT/'scene_groups.csv';CHANGES=REPORT/'strict_transfer_changes.csv';ACTIONS=REPORT/'scene_group_actions.csv';SPLITS={'train','val','test'};LOCK=threading.RLock()
def read(path):
 with path.open(encoding='utf-8',newline='') as f:r=csv.DictReader(f);return list(r.fieldnames or []),list(r)
def write(path,fields,rows):
 with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',delete=False,dir=path.parent) as f:t=Path(f.name);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
 try:os.replace(t,path)
 finally:t.unlink(missing_ok=True)
def label(s,i):return DATA/'labels'/s/(Path(i).stem+'.txt')
def image(s,i):return DATA/'images'/s/i
def boxes(path):
 out=[]
 for line in path.read_text(encoding='utf-8').splitlines():
  if line.strip():
   c,x,y,w,h=line.split();out.append({'class_id':int(c),'x':float(x),'y':float(y),'width':float(w),'height':float(h)})
 return out
def serial(boxes_):return ''.join(f"{b['class_id']} {b['x']:.6f} {b['y']:.6f} {b['width']:.6f} {b['height']:.6f}\n" for b in boxes_)
def atomic(path,text):
 fd,t=tempfile.mkstemp(prefix='.label.',dir=path.parent)
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
  os.replace(t,path)
 finally:
  if os.path.exists(t):os.unlink(t)
def iou(a,b):
 ax1,ay1,ax2,ay2=a['x']-a['width']/2,a['y']-a['height']/2,a['x']+a['width']/2,a['y']+a['height']/2;bx1,by1,bx2,by2=b['x']-b['width']/2,b['y']-b['height']/2,b['x']+b['width']/2,b['y']+b['height']/2
 z=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1));u=a['width']*a['height']+b['width']*b['height']-z;return z/u if u else 0
def manual():return {(r['split'],r['image']) for r in read(PREP/'geometry_changes.csv')[1]}
def state(index):
 _,gs=read(GROUPS);index=max(0,min(index,len(gs)-1));g=gs[index];_,changes=read(CHANGES);c=[r for r in changes if r['group_id']==g['group_id'] and r['status'] in {'dry_run_high','applied_group'}]
 return {'index':index,'total':len(gs),'group':g,'candidates':c,'target_boxes':boxes(label(g['target_split'],g['target_image'])),'reference_boxes':boxes(label(g['reference_split'],g['reference_image'])),'can_apply':g['status']!='applied' and bool(c),'can_undo':g['status']=='applied'}
def render(index):
 s=state(index);g=s['group'];items=[(g['reference_split'],g['reference_image'],s['reference_boxes'],[],'Эталон: ручные рамки'),(g['target_split'],g['target_image'],s['target_boxes'],[json.loads(r['new_box']) for r in s['candidates']],'Цель: зелёный — было; оранжевый — перенос')];out=Image.new('RGB',(1100,620),'#101820')
 for n,(sp,im,old,new,title) in enumerate(items):
  pic=Image.open(image(sp,im)).convert('RGB');pic.thumbnail((530,540));d=ImageDraw.Draw(pic);w,h=pic.size
  for collection,color in ((old,'#42d37b'),(new,'#ff9f1a')):
   for b in collection:d.rectangle(((b['x']-b['width']/2)*w,(b['y']-b['height']/2)*h,(b['x']+b['width']/2)*w,(b['y']+b['height']/2)*h),outline=color,width=3)
  out.paste(pic,(15+n*550,35));ImageDraw.Draw(out).text((15+n*550,8),title,fill='white');ImageDraw.Draw(out).text((15+n*550,580),f'{sp}/{im}',fill='white')
 import io;b=io.BytesIO();out.save(b,'JPEG',quality=90);return b.getvalue()
def action(group_id,undo=False):
 with LOCK:
  gf,gs=read(GROUPS);g=next((x for x in gs if x['group_id']==group_id),None)
  if not g:raise ValueError('Группа не найдена')
  if (g['target_split'],g['target_image']) in manual():raise ValueError('Ручной эталонный кадр защищён')
  cf,cs=read(CHANGES);related=[x for x in cs if x['group_id']==group_id]
  lf=label(g['target_split'],g['target_image']);before=lf.read_text(encoding='utf-8')
  af,actions=read(ACTIONS) if ACTIONS.exists() else (['timestamp','group_id','action','split','image','before','after'],[])
  if undo:
   last=next((x for x in reversed(actions) if x['group_id']==group_id and x['action']=='apply'),None)
   if not last:raise ValueError('Для группы нет применённого состояния')
   if before!=last['after']:raise ValueError('Нельзя безопасно откатить: label изменён после группового применения')
   atomic(lf,last['before']);g['status']='dry_run';
   for x in related:
    if x['status']=='applied_group':x['status']='dry_run_high'
   after=last['before'];verb='undo'
  else:
   if g['status']=='applied':raise ValueError('Группа уже применена')
   current=boxes(lf);added=[]
   for x in related:
    if x['status']!='dry_run_high':continue
    b=json.loads(x['new_box'])
    if any(iou(b,old)>=.80 for old in current):x['status']='skipped_duplicate';continue
    current.append(b);added.append(b);x['status']='applied_group'
   after=serial(current);atomic(lf,after);g['status']='applied';g['applied_boxes']=str(len(added));verb='apply'
  write(GROUPS,gf,gs);write(CHANGES,cf,cs);actions.append({'timestamp':datetime.now(timezone.utc).isoformat(),'group_id':group_id,'action':verb,'split':g['target_split'],'image':g['target_image'],'before':before,'after':after});write(ACTIONS,af,actions)
  return {'ok':True,'action':verb,'group_id':group_id,'affected_images':1,'added_boxes':g.get('applied_boxes','0')}
PAGE='''<!doctype html><meta charset="utf-8"><title>V3 — группы похожих сцен</title><style>body{margin:0;background:#10151d;color:#eff6ff;font:15px system-ui}header,main{padding:16px;max-width:1140px;margin:auto}header{display:flex;gap:18px;align-items:center;flex-wrap:wrap;background:#17212d}h1{font-size:20px;margin:0}img{width:100%;border-radius:10px;background:#000}.panel{background:#17212d;padding:14px;border-radius:10px;margin-top:12px;white-space:pre-wrap}button{margin:4px;padding:10px 14px;border:0;border-radius:7px;color:#fff;background:#2e4760;font:inherit;cursor:pointer}button.ok{background:#157f59}button.warn{background:#914046}.muted{color:#b6c7d7}</style><body><header><h1>V3 — группы похожих сцен</h1><b id=p>Загрузка…</b><span id=s></span></header><main><img id=i><div class=panel id=m></div><div class=panel><button id=prev>Назад</button><button id=next>Далее</button><button class=ok id=apply>Применить рамки к похожему кадру</button><button class=warn id=undo>Откатить группу</button><p class=muted>Перед применением рамки уже показаны оранжевым. Операция затрагивает ровно один целевой кадр в этой строгой группе и сохраняет снимок до/после; откат доступен, пока label не изменён после применения.</p></div></main><script>let x;const q=s=>document.querySelector(s);function msg(t,e){q('#s').textContent=t;q('#s').style.color=e?'#ff9c9c':'#9ae8b8'}function load(n){fetch('/api/state?index='+n).then(r=>r.json()).then(z=>{if(z.error)throw Error(z.error);x=z;let g=z.group;q('#p').textContent=`${z.index+1}/${z.total}: ${g.group_id}`;q('#m').textContent=`Эталон: ${g.reference_split}/${g.reference_image}\nПохожий кадр: ${g.target_split}/${g.target_image}\nПереносимых рамок: ${z.candidates.length}\nГистограмма: ${g.histogram_similarity}; RANSAC inliers: ${g.homography_inliers}; ratio: ${g.inlier_ratio}\nСтатус: ${g.status}`;q('#i').src='/render?index='+z.index+'&v='+Date.now();q('#apply').disabled=!z.can_apply;q('#undo').disabled=!z.can_undo}).catch(e=>msg(e.message,true))}function post(url){fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({group_id:x.group.group_id})}).then(r=>r.json()).then(z=>{if(z.error)throw Error(z.error);msg(`${z.action}: затронуто изображений ${z.affected_images}, добавлено рамок ${z.added_boxes}`);load(x.index)}).catch(e=>msg(e.message,true))}q('#prev').onclick=()=>load(Math.max(0,x.index-1));q('#next').onclick=()=>load(Math.min(x.total-1,x.index+1));q('#apply').onclick=()=>post('/api/apply');q('#undo').onclick=()=>post('/api/undo');load(0)</script>'''
class H(BaseHTTPRequestHandler):
 def sendj(self,x,status=200):
  b=json.dumps(x,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json;charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def do_GET(self):
  p=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(p.query)
  try:
   if p.path=='/':b=PAGE.encode();self.send_response(200);self.send_header('Content-Type','text/html;charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
   elif p.path=='/api/state':self.sendj(state(int(q.get('index',['0'])[0])))
   elif p.path=='/render':b=render(int(q.get('index',['0'])[0]));self.send_response(200);self.send_header('Content-Type','image/jpeg');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
   else:self.send_error(HTTPStatus.NOT_FOUND)
  except Exception as e:self.sendj({'error':str(e)},400)
 def do_POST(self):
  try:
   p=urllib.parse.urlparse(self.path);n=int(self.headers.get('Content-Length','0'));body=json.loads(self.rfile.read(n));self.sendj(action(body.get('group_id',''),p.path=='/api/undo') if p.path in {'/api/apply','/api/undo'} else {'error':'not found'},200 if p.path in {'/api/apply','/api/undo'} else 404)
  except Exception as e:self.sendj({'error':str(e)},400)
 def log_message(self,*x):pass
def main():
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8095);a=p.parse_args();ThreadingHTTPServer((a.host,a.port),H).serve_forever()
if __name__=='__main__':main()
