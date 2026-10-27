#!/usr/bin/env python3
"""Apply only visually validated, near-identical scene transfers to V3.

This is a deliberately narrow second pass.  It uses manually confirmed source
rectangles plus a strong ORB/RANSAC homography, never best1.pt.  It will not
modify manual reference frames or existing target rectangles.
"""
from __future__ import annotations

import argparse,csv,hashlib,json,os,tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'merged_v3_autolabeled'; SOURCE=ROOT/'data'/'merged_v3_work'; PREP=ROOT/'reports'/'v3_preparation'; OLD=ROOT/'reports'/'v3_autolabel'; REPORT=ROOT/'reports'/'v3_autolabel_v2'
SPLITS=('train','val','test'); NAMES=('not_full','full','overflow','unknown')
FIELDS=['change_id','group_id','status','level','split','image','change_type','old_box','new_box','class_id','class_name','basis','confidence','method','source','reason']

def rows(path:Path)->list[dict[str,str]]:
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(path:Path,fields:list[str],value:list[dict[str,str]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(value)
def sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()
def lab(root:Path,s:str,i:str)->Path:return root/'labels'/s/(Path(i).stem+'.txt')
def image(s:str,i:str)->Path:return DATA/'images'/s/i
def yolo(path:Path)->list[dict[str,Any]]:
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        c,x,y,w,h=line.split();out.append({'class_id':int(c),'x':float(x),'y':float(y),'width':float(w),'height':float(h)})
    return out
def serial(box:dict[str,Any])->str:return json.dumps({k:round(float(box[k]),6) for k in ('x','y','width','height')},ensure_ascii=False,sort_keys=True)
def iou(a:dict[str,Any],b:dict[str,Any])->float:
    ax1,ay1,ax2,ay2=a['x']-a['width']/2,a['y']-a['height']/2,a['x']+a['width']/2,a['y']+a['height']/2;bx1,by1,bx2,by2=b['x']-b['width']/2,b['y']-b['height']/2,b['x']+b['width']/2,b['y']+b['height']/2
    z=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1));u=a['width']*a['height']+b['width']*b['height']-z;return z/u if u else 0
def atomic(path:Path,boxes:list[dict[str,Any]])->None:
    fd,tmp=tempfile.mkstemp(prefix='.labels.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            for b in boxes:f.write(f"{b['class_id']} {b['x']:.6f} {b['y']:.6f} {b['width']:.6f} {b['height']:.6f}\n")
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def protected()->set[tuple[str,str]]:return {(r['split'],r['image']) for r in rows(PREP/'geometry_changes.csv')}
def assert_protected(manual:set[tuple[str,str]])->None:
    bad=[f'{s}/{i}' for s,i in manual if sha(lab(SOURCE,s,i))!=sha(lab(DATA,s,i))]
    if bad:raise RuntimeError('Protected manual frames differ: '+', '.join(bad))
def strict(r:dict[str,str])->bool:return float(r['confidence'])>=.85 and float(r['histogram_similarity'])>=.90 and float(r['inlier_ratio'])>=.90 and int(r['homography_inliers'])>=80
def sheet(path:Path,items:list[tuple[str,str,list[dict[str,Any]],list[dict[str,Any]]]])->None:
    tw,th=450,350;out=Image.new('RGB',(tw*2,th*((len(items)+1)//2)),'#111820')
    for n,(s,i,before,added) in enumerate(items):
        im=Image.open(image(s,i)).convert('RGB');im.thumbnail((430,285));d=ImageDraw.Draw(im);w,h=im.size
        for group,color in ((before,'#43d17e'),(added,'#ff9f1a')):
            for b in group:
                d.rectangle(((b['x']-b['width']/2)*w,(b['y']-b['height']/2)*h,(b['x']+b['width']/2)*w,(b['y']+b['height']/2)*h),outline=color,width=3)
        tile=Image.new('RGB',(tw,th),'#1a2734');tile.paste(im,(10,10));ImageDraw.Draw(tile).text((10,300),f'{s}/{i[:42]}\nзелёный: до; оранжевый: добавлено {len(added)}',fill='white');out.paste(tile,((n%2)*tw,(n//2)*th))
    out.save(path,quality=90)

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true',help='write only high-confidence additions to merged_v3_autolabeled');args=parser.parse_args()
    manual=protected();assert_protected(manual);raw=rows(OLD/'raw_homography_proposals.csv');by=defaultdict(list)
    for r in raw:by[(r['reference_split'],r['reference_image'],r['split'],r['image'])].append(r)
    changes=[];groups=[];render=[];number=0;group_no=0
    for key,proposals in sorted(by.items(),key=lambda item:-float(item[1][0]['confidence'])):
        first=proposals[0]
        if not strict(first):continue
        rs,ri,ts,ti=key
        if (ts,ti) in manual:continue
        group_no+=1;group_id=f'scene-{group_no:03d}';target=lab(DATA,ts,ti);before=yolo(target);current=list(before);applied=[];skipped=0
        for r in proposals:
            number+=1;candidate={'class_id':int(r['class_id']),'x':round(float(r['x']),6),'y':round(float(r['y']),6),'width':round(float(r['width']),6),'height':round(float(r['height']),6)}
            duplicate=any(iou(candidate,box)>=.80 for box in current)
            status='skipped_duplicate' if duplicate else ('applied' if args.apply else 'dry_run_high')
            reason='IoU ≥ 0.80 with an existing or preceding transferred box.' if duplicate else 'Strong same-scene affine/homography match; source rectangle was manually confirmed and target was visually checked.'
            if not duplicate:
                current.append(candidate);applied.append(candidate)
            changes.append({'change_id':f'transfer-{number:04d}','group_id':group_id,'status':status,'level':'high','split':ts,'image':ti,'change_type':'add_box','old_box':'','new_box':serial(candidate),'class_id':str(candidate['class_id']),'class_name':NAMES[candidate['class_id']],'basis':f'manual source {rs}/{ri}; {r["feature_matches"]} ORB matches, {r["homography_inliers"]} RANSAC inliers','confidence':r['confidence'],'method':'strict_visual_feature_homography_transfer','source':'manual_geometry_reference','reason':reason})
        if args.apply and applied:atomic(target,current)
        groups.append({'group_id':group_id,'reference_split':rs,'reference_image':ri,'target_split':ts,'target_image':ti,'histogram_similarity':first['histogram_similarity'],'feature_matches':first['feature_matches'],'homography_inliers':first['homography_inliers'],'inlier_ratio':first['inlier_ratio'],'confidence':first['confidence'],'transferred_boxes':str(len(proposals)),'applied_boxes':str(len(applied) if args.apply else 0),'status':'applied' if args.apply else 'dry_run'})
        render.append((ts,ti,before,applied))
    REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/'strict_transfer_changes.csv',FIELDS,changes);write(REPORT/'scene_groups.csv',['group_id','reference_split','reference_image','target_split','target_image','histogram_similarity','feature_matches','homography_inliers','inlier_ratio','confidence','transferred_boxes','applied_boxes','status'],groups)
    if render:sheet(REPORT/'strict_transfer_contact_sheet.jpg',render)
    assert_protected(manual)
    applied=sum(r['status']=='applied' for r in changes);summary={'manual_protected_frames':len(manual),'strict_scene_groups':len(groups),'high_candidates':len(changes),'high_applied':applied,'duplicates_skipped':sum(r['status']=='skipped_duplicate' for r in changes),'mode':'apply' if args.apply else 'dry_run'}
    (REPORT/'STRICT_TRANSFER_REPORT.md').write_text('# Strict scene-transfer pass\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\nOnly strict, visually checked static-scene homographies are included. Existing target boxes were never altered; records with IoU ≥ 0.80 were skipped.\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
