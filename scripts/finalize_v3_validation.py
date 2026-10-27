#!/usr/bin/env python3
"""Final v3 QA. It never trains, archives, or edits data outside v3 work."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "merged_v3_work"
REPORTS = ROOT / "reports" / "v3_preparation"
QUEUE = REPORTS / "manual_review_queue.csv"
NAMES = ("not_full", "full", "overflow", "unknown")
SPLITS = ("train", "val", "test")


def boxes(path: Path):
    result=[]
    for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            raise ValueError(f'{path}:{number}: empty line')
        fields=line.split()
        if len(fields)!=5: raise ValueError(f'{path}:{number}: invalid YOLO field count')
        cls=int(fields[0]); x,y,w,h=map(float,fields[1:])
        if cls not in range(4) or not(0<=x<=1 and 0<w<=1 and 0<h<=1): raise ValueError(f'{path}:{number}: invalid class/coordinates')
        if x-w/2<0 or x+w/2>1 or y-h/2<0 or y+h/2>1: raise ValueError(f'{path}:{number}: bbox outside image')
        result.append((cls,x,y,w,h))
    if len(set(result))!=len(result): raise ValueError(f'{path}: duplicate bbox')
    return result


def write_csv(path, fields, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def draw_page(items, destination):
    tiles=[]
    for image_path, current in items:
        with Image.open(image_path) as opened: image=opened.convert('RGB')
        draw=ImageDraw.Draw(image)
        for cls,x,y,w,h in current:
            left=(x-w/2)*image.width;top=(y-h/2)*image.height;right=(x+w/2)*image.width;bottom=(y+h/2)*image.height
            color=('#37c878','#42a7ff','#f05a4f','#bd83f5')[cls]
            draw.rectangle((left,top,right,bottom),outline=color,width=max(2,image.width//300));draw.text((left+2,max(2,top-14)),NAMES[cls],fill=color,stroke_width=1,stroke_fill='black')
        image.thumbnail((420,310));tile=Image.new('RGB',(430,338),'#111');tile.paste(image,((430-image.width)//2,5));ImageDraw.Draw(tile).text((5,318),image_path.name[:64],fill='white');tiles.append(tile)
    columns=3;rows=(len(tiles)+columns-1)//columns;sheet=Image.new('RGB',(columns*430,rows*338),'#111')
    for index,tile in enumerate(tiles):sheet.paste(tile,((index%columns)*430,(index//columns)*338))
    sheet.save(destination,quality=90)


def main():
    queue=list(csv.DictReader(QUEUE.open(encoding='utf-8')))
    pending=[row for row in queue if row['status']!='reviewed']
    if pending: raise SystemExit(f'Manual review is incomplete: {len(pending)} queue items remain')
    errors=[];stats=[];samples={name:[] for name in NAMES};digests={};all_images=[]
    for split in SPLITS:
        images=sorted((DATA/'images'/split).iterdir());labels=sorted((DATA/'labels'/split).glob('*.txt'));counts=Counter();empty=0
        if len(images)!=len(labels):errors.append(f'{split}: image/label count mismatch {len(images)}/{len(labels)}')
        for label in labels:
            image=next((p for p in images if p.stem==label.stem),None)
            if image is None:errors.append(f'missing image for {label.name}');continue
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in digests and digests[digest] != split:errors.append(f'duplicate image across splits: {image.name}')
            digests[digest]=split;all_images.append((image,label))
            try: current=boxes(label)
            except Exception as exc:errors.append(str(exc));continue
            if not current:empty+=1
            for box in current:
                counts[box[0]]+=1;samples[NAMES[box[0]]].append((image,current))
        stats.append({'split':split,'images':len(images),'empty_labels':empty,'objects':sum(counts.values()),**{name:counts[index] for index,name in enumerate(NAMES)}})
    total={'split':'total','images':sum(x['images'] for x in stats),'empty_labels':sum(x['empty_labels'] for x in stats),'objects':sum(x['objects'] for x in stats)};total.update({name:sum(x[name] for x in stats) for name in NAMES});stats.append(total)
    if errors: raise SystemExit('QA failed:\n'+'\n'.join(errors[:30]))
    write_csv(REPORTS/'class_statistics.csv',['split','images','empty_labels','objects',*NAMES],stats)
    sheets=REPORTS/'contact_sheets';sheets.mkdir(exist_ok=True)
    for name,items in samples.items():
        unique=list({str(image):(image,current) for image,current in items}.values())
        selected=random.Random(20260920).sample(unique,min(24,len(unique)))
        if selected:draw_page(selected,sheets/f'{name}_random.jpg')
        if name in {'overflow','unknown'}:
            for start in range(0,len(unique),24):draw_page(unique[start:start+24],sheets/f'{name}_all_{start//24+1:03d}.jpg')
    report=['# V3 final quality report','','- Manual review queue: complete.','- Training and ZIP creation were not performed.','',f'- Images: **{total["images"]}**; objects: **{total["objects"]}**; empty labels: **{total["empty_labels"]}**.','- Image–label correspondence, YOLO format, IDs 0–3, coordinate ranges, duplicate boxes and cross-split image duplicates: passed.','','## Class totals','', '| class | objects |','| --- | ---: |',*[f'| {name} | {total[name]} |' for name in NAMES],'','Contact sheets: `reports/v3_preparation/contact_sheets/`.']
    (REPORTS/'final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(total,ensure_ascii=False))


if __name__=='__main__':main()
