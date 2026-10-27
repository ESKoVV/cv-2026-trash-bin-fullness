#!/usr/bin/env python3
"""Read-only final quality audit for ``merged_v3_autolabeled``.

The audit deliberately does not infer labels or fix visual objects.  It
records deterministic structural problems, known unresolved visual work, split
leakage evidence, and audit contact sheets.  This prevents a clean exit code
from being mistaken for a visual sign-off.
"""
from __future__ import annotations

import csv, hashlib, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, UnidentifiedImageError

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'/'merged_v3_autolabeled';REPORT=ROOT/'reports'/'v3_final_audit';PREP=ROOT/'reports'/'v3_preparation';AUTO=ROOT/'reports'/'v3_autolabel';AUTO2=ROOT/'reports'/'v3_autolabel_v2'
SPLITS=('train','val','test');NAMES=('not_full','full','overflow','unknown');EXT={'.jpg','.jpeg','.png','.webp'}

def csv_rows(path:Path)->list[dict[str,str]]:
    if not path.exists():return []
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
def csv_write(path:Path,fields:list[str],rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()
def lab(s:str,image:Path)->Path:return DATA/'labels'/s/f'{image.stem}.txt'
def parse_label(path:Path)->tuple[list[dict[str,Any]],list[str]]:
    boxes=[];issues=[]
    for line_no,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        p=line.split()
        if len(p)!=5:issues.append(f'line_{line_no}_field_count');continue
        try:c=int(p[0]);x,y,w,h=map(float,p[1:])
        except ValueError:issues.append(f'line_{line_no}_non_numeric');continue
        eps=1e-6
        if c not in range(4):issues.append(f'line_{line_no}_class_id')
        if not(0<w<=1 and 0<h<=1 and -eps<=x-w/2 and x+w/2<=1+eps and -eps<=y-h/2 and y+h/2<=1+eps):issues.append(f'line_{line_no}_coordinates')
        boxes.append({'class_id':c,'x':x,'y':y,'width':w,'height':h})
    return boxes,issues
def iou(a:dict[str,Any],b:dict[str,Any])->float:
    ax1,ay1,ax2,ay2=a['x']-a['width']/2,a['y']-a['height']/2,a['x']+a['width']/2,a['y']+a['height']/2;bx1,by1,bx2,by2=b['x']-b['width']/2,b['y']-b['height']/2,b['x']+b['width']/2,b['y']+b['height']/2
    area=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1));union=a['width']*a['height']+b['width']*b['height']-area;return area/union if union else 0
def dhash(path:Path)->int:
    with Image.open(path) as image:
        im=image.convert('L').resize((9,8));pixels=list(im.getdata())
    bits=0
    for y in range(8):
        for x in range(8):bits=(bits<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    return bits
def hamming(a:int,b:int)->int:return (a^b).bit_count()
def draw_sheet(path:Path,items:list[tuple[str,Path,list[dict[str,Any]],str]],columns:int=4)->None:
    tw,th=330,270;out=Image.new('RGB',(tw*columns,th*max(1,math.ceil(len(items)/columns))),'#101820')
    for n,(split,image,boxes,caption) in enumerate(items):
        im=Image.open(image).convert('RGB');im.thumbnail((310,215));d=ImageDraw.Draw(im);w,h=im.size
        for box in boxes:
            color=('#42d37b','#4da8ff','#ff9f1a','#c184ff')[box['class_id']] if box['class_id'] in range(4) else '#f00'
            d.rectangle(((box['x']-box['width']/2)*w,(box['y']-box['height']/2)*h,(box['x']+box['width']/2)*w,(box['y']+box['height']/2)*h),outline=color,width=2)
        tile=Image.new('RGB',(tw,th),'#172534');tile.paste(im,(10,10));ImageDraw.Draw(tile).text((10,232),f'{split}/{image.name[:30]}\n{caption}',fill='white');out.paste(tile,((n%columns)*tw,(n//columns)*th))
    out.save(path,quality=88)

def main()->None:
    REPORT.mkdir(parents=True,exist_ok=True);sheets=REPORT/'contact_sheets';sheets.mkdir(exist_ok=True)
    yaml_text=(DATA/'data.yaml').read_text(encoding='utf-8') if (DATA/'data.yaml').exists() else ''
    expected_yaml=('train: images/train','val: images/val','test: images/test','nc: 4','0: not_full','1: full','2: overflow','3: unknown')
    images:dict[tuple[str,str],Path]={};labels={};integrity={'data_yaml':{'exists':bool(yaml_text),'expected_mapping':all(value in yaml_text for value in expected_yaml),'path':str(DATA/'data.yaml')},'splits':{},'missing_labels':[],'orphan_labels':[],'label_issues':[],'corrupt_images':[],'duplicates_exact_cross_split':[],'near_duplicates_cross_split':[]}
    stats=Counter();by_split=Counter();hashes=defaultdict(list);phashes=[];hard_suspects=[]
    for split in SPLITS:
        image_dir=DATA/'images'/split;label_dir=DATA/'labels'/split;files=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXT);label_files={p.stem:p for p in label_dir.glob('*.txt')};integrity['splits'][split]={'images':len(files),'labels':len(label_files)}
        for image in files:
            key=(split,image.name);images[key]=image;path=lab(split,image)
            if not path.exists():integrity['missing_labels'].append(f'{split}/{image.name}'); boxes=[]
            else:boxes,issues=parse_label(path);labels[key]=boxes;integrity['label_issues'].extend({'split':split,'image':image.name,'issue':issue} for issue in issues)
            try:
                with Image.open(image) as verify:verify.verify()
            except (UnidentifiedImageError,OSError):integrity['corrupt_images'].append(f'{split}/{image.name}')
            digest=sha(image);hashes[digest].append(key);phashes.append((split,image.name,dhash(image)))
            by_split[split]+=len(boxes)
            for box in boxes:
                stats[(split,box['class_id'])]+=1;stats[('total',box['class_id'])]+=1
            for a in range(len(boxes)):
                for b in range(a+1,len(boxes)):
                    overlap=iou(boxes[a],boxes[b])
                    if overlap>=.95:hard_suspects.append((split,image.name,'near_duplicate_boxes',round(overlap,4)))
                    elif overlap>=.75:hard_suspects.append((split,image.name,'strong_box_overlap',round(overlap,4)))
        for stem in label_files:
            if not any(p.stem==stem for p in files):integrity['orphan_labels'].append(f'{split}/{label_files[stem].name}')
    for value,items in hashes.items():
        if len({x[0] for x in items})>1:integrity['duplicates_exact_cross_split'].append({'sha256':value,'items':['/'.join(x) for x in items]})
    # dHash flags only high-confidence, almost identical images. More permissive
    # scene similarity is supplied by previously verified ORB/RANSAC matches.
    for i,(sa,ia,ha) in enumerate(phashes):
        for sb,ib,hb in phashes[i+1:]:
            if sa!=sb and hamming(ha,hb)<=3:integrity['near_duplicates_cross_split'].append({'left':f'{sa}/{ia}','right':f'{sb}/{ib}','distance':hamming(ha,hb),'source':'dhash'})
    # Existing strict homographies are stronger evidence of a camera/sequence leak.
    leakage=[]
    for row in csv_rows(AUTO2/'scene_groups.csv'):
        if row['reference_split']!=row['target_split']:
            leakage.append({'severity':'critical','left':f"{row['reference_split']}/{row['reference_image']}",'right':f"{row['target_split']}/{row['target_image']}",'evidence':'strict_manual_reference_homography','histogram_similarity':row['histogram_similarity'],'inlier_ratio':row['inlier_ratio'],'homography_inliers':row['homography_inliers'],'recommendation':'Move the whole same-camera sequence to one split, then rebuild split statistics.'})
    for row in integrity['duplicates_exact_cross_split']:
        leakage.append({'severity':'critical','left':row['items'][0],'right':' | '.join(row['items'][1:]),'evidence':'exact_sha256_duplicate','histogram_similarity':'','inlier_ratio':'','homography_inliers':'','recommendation':'Keep all copies in one split or remove duplicates.'})
    for row in integrity['near_duplicates_cross_split']:
        leakage.append({'severity':'review','left':row['left'],'right':row['right'],'evidence':'dhash_near_duplicate','histogram_similarity':'','inlier_ratio':'','homography_inliers':'','recommendation':'Visually verify sequence/camera and keep it in one split.'})
    csv_write(REPORT/'split_leakage.csv',['severity','left','right','evidence','histogram_similarity','inlier_ratio','homography_inliers','recommendation'],leakage)
    class_rows=[]
    for split in (*SPLITS,'total'):
        for class_id,name in enumerate(NAMES):class_rows.append({'split':split,'class_id':class_id,'class_name':name,'objects':stats[(split,class_id)],'images':sum(1 for key,boxes in labels.items() if (split=='total' or key[0]==split) and any(b['class_id']==class_id for b in boxes))})
    csv_write(REPORT/'class_statistics.csv',['split','class_id','class_name','objects','images'],class_rows)
    # Aggregate unresolved visual work by frame, rather than pretending it was
    # audited.  This preserves pending work and gives strict transfers priority.
    remaining=defaultdict(list)
    for row in csv_rows(AUTO/'manual_check_queue.csv'):
        if row.get('status')=='pending':remaining[(row['split'],row['image'])].append(f"{row.get('source','manual_queue')}: {row.get('reason','pending visual check')}")
    for row in csv_rows(AUTO2/'strict_transfer_changes.csv'):
        if row['status'] in {'dry_run_high','skipped_duplicate'}:remaining[(row['split'],row['image'])].append(f"strict_transfer_{row['status']}: {row['change_id']}")
    for split,image_name,issue,value in hard_suspects:remaining[(split,image_name)].append(f'geometry_heuristic: {issue} ({value})')
    review_rows=[]
    for number,((split,image_name),reasons) in enumerate(sorted(remaining.items()),1):review_rows.append({'review_id':f'final-{number:04d}','split':split,'image':image_name,'priority':'high' if any('strict_transfer' in x for x in reasons) else 'medium','reason':' | '.join(dict.fromkeys(reasons)),'status':'pending'})
    csv_write(REPORT/'remaining_review.csv',['review_id','split','image','priority','reason','status'],review_rows)
    for name,fields in [('missing_boxes_added.csv',['timestamp','split','image','class_id','class_name','box','basis']),('boxes_corrected.csv',['timestamp','split','image','old_box','new_box','basis']),('classes_corrected.csv',['timestamp','split','image','old_class','new_class','basis']),('manual_overrides_changed.csv',['timestamp','split','image','change','basis'])]:csv_write(REPORT/name,fields,[])
    # Control and review sheets: no annotation is altered by this audit.
    random.seed(20260920);control=random.sample(list(images),min(100,len(images)));draw_sheet(sheets/'control_random_100.jpg',[(s,images[(s,i)],labels.get((s,i),[]),'контроль после аудита') for s,i in control],4)
    review_items=[]
    for row in review_rows[:100]:
        key=(row['split'],row['image']);review_items.append((key[0],images[key],labels.get(key,[]),row['priority']))
    if review_items:draw_sheet(sheets/'remaining_review_first_100.jpg',review_items,4)
    (sheets/'README.md').write_text('В этом аудите новые рамки, исправления геометрии и классов не применялись: соответствующие листы отсутствуют намеренно. `control_random_100.jpg` — случайная контрольная выборка; `remaining_review_first_100.jpg` — первые 100 нерешённых кадров.\n',encoding='utf-8')
    integrity['images_total']=len(images);integrity['objects_total']=sum(by_split.values());integrity['objects_by_split']=dict(by_split);integrity['overlap_suspects']=len(hard_suspects);integrity['box_overlap_suspects']=[{'split':s,'image':i,'issue':issue,'iou':value} for s,i,issue,value in hard_suspects];integrity['remaining_review_frames']=len(review_rows);integrity['strict_high_candidates_pending']=sum(1 for r in csv_rows(AUTO2/'strict_transfer_changes.csv') if r['status']=='dry_run_high');integrity['ready']=False;integrity['verdict']='NOT READY';integrity['blocking_reasons']=['Visual completeness cannot be certified: pending final review entries remain.','High-confidence scene-transfer candidates have not been manually approved/applied.','Cross-split same-scene leakage is present.','No human visual sign-off exists for every frame/box/class.']
    (REPORT/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit=f'''# Final audit — V3

## Verdict: NOT READY

The dataset must **not** be used for training yet. This verdict is based on the
actual audit evidence, not merely on a successful script run.

## Structural checks

- Images: {len(images)}; label objects: {sum(by_split.values())}.
- `data.yaml` exists and maps all four expected class IDs: {integrity['data_yaml']['expected_mapping']}.
- Missing labels: {len(integrity['missing_labels'])}; orphan labels: {len(integrity['orphan_labels'])}; corrupt images: {len(integrity['corrupt_images'])}; invalid YOLO records: {len(integrity['label_issues'])}.
- Exact duplicate boxes / strong overlaps flagged: {len(hard_suspects)} (heuristic flags, not automatic corrections).

## Blocking visual work

- Remaining unique frames: **{len(review_rows)}**.
- Pending strict same-scene transfer candidates: **{integrity['strict_high_candidates_pending']}**. They include background-container rows from manually corrected reference scenes, but are not treated as labels until a person approves them.
- No boxes/classes/manual overrides were changed by this audit. The four corresponding CSVs are intentionally header-only.

## Split integrity

- Cross-split exact duplicates: {len(integrity['duplicates_exact_cross_split'])}.
- Cross-split near-duplicate dHash flags: {len(integrity['near_duplicates_cross_split'])}.
- High-confidence cross-split same-scene leaks: {sum(r['severity']=='critical' and r['evidence']=='strict_manual_reference_homography' for r in leakage)}.

At least one strict train→test same-camera transfer is present in `split_leakage.csv`. Test results would therefore be optimistic until complete camera sequences are kept in one split.

## Shortest safe path to readiness

1. Review and either accept/correct/reject the 59 strict scene-transfer boxes in the group UI; do not invent uncertain far objects.
2. Resolve the remaining visual-review frames, prioritising yellow-container groups and all geometry flags.
3. Re-split complete duplicate/near-duplicate camera sequences so they do not cross train/val/test.
4. Re-run this audit. Only an empty `remaining_review.csv`, a clean split-leakage review and a visual sign-off can change the verdict.
'''
    (REPORT/'FINAL_AUDIT.md').write_text(audit,encoding='utf-8')
    print(json.dumps({'verdict':'NOT READY','images':len(images),'objects':sum(by_split.values()),'remaining_frames':len(review_rows),'critical_split_leaks':sum(r['severity']=='critical' for r in leakage),'invalid_records':len(integrity['label_issues'])},ensure_ascii=False))
if __name__=='__main__':main()
