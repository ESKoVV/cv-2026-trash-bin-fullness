#!/usr/bin/env python3
"""Read-only zero-shot container proposals using YOLO-World.

The detector identifies container geometry only. Fill level is intentionally
not inferred: every proposal is emitted as class `unknown` for later review.
"""
from __future__ import annotations
import argparse,csv
from pathlib import Path
from ultralytics import YOLO

EXT={'.jpg','.jpeg','.png','.webp'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=8);args=p.parse_args()
 entries=[(split,path) for split in ('train','val','test') for path in sorted((args.images/split).iterdir()) if path.suffix.lower() in EXT]
 model=YOLO(str(args.weights));model.set_classes(['trash bin','garbage container','dumpster','waste bin','wheelie bin'])
 fields=['split','image','class_id','class_name','x','y','width','height','confidence','method']
 with args.output.open('w',encoding='utf-8',newline='') as h:
  w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
  for start in range(0,len(entries),args.batch):
   batch=entries[start:start+args.batch];results=model([str(path) for _,path in batch],imgsz=960,conf=.05,iou=.55,device='cpu',verbose=False)
   for (split,path),result in zip(batch,results,strict=True):
    if result.boxes is None:continue
    for box in result.boxes:
     x1,y1,x2,y2=map(float,box.xyxy[0].tolist());width,height=result.orig_shape[1],result.orig_shape[0]
     w.writerow({'split':split,'image':path.name,'class_id':3,'class_name':'unknown','x':round((x1+x2)/2/width,6),'y':round((y1+y2)/2/height,6),'width':round((x2-x1)/width,6),'height':round((y2-y1)/height,6),'confidence':round(float(box.conf.item()),6),'method':'zero_shot_yolo_world_container_detection'})
   h.flush()
   print(f'processed {min(start+len(batch),len(entries))}/{len(entries)}',flush=True)
if __name__=='__main__':main()
