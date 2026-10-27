#!/usr/bin/env python3
"""Measure tiled Grounding-DINO geometry against manually confirmed V3 frames.

The script is intentionally read-only.  It needs an environment with
``transformers`` and ``torch``; it never writes a label file or loads best1.pt.
"""
from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

PROMPTS = ["garbage bin", "trash bin", "waste container", "dumpster", "wheelie bin", "recycling container", "refuse container"]


def iou(a: list[float], b: list[float]) -> float:
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0.0,x2-x1)*max(0.0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0


def nms(rows: list[dict], threshold: float) -> list[dict]:
    selected=[]
    for row in sorted(rows,key=lambda row:-row['confidence']):
        if all(iou(row['xyxy'], previous['xyxy']) < threshold for previous in selected): selected.append(row)
    return selected


def tiles(width: int, height: int, size: int, overlap: float) -> list[tuple[int,int,int,int]]:
    """A full-frame pass plus native-resolution overlapping tiles when needed."""
    result=[(0,0,width,height)]
    if width<=size and height<=size: return result
    step=max(1,round(size*(1-overlap)))
    xs=list(range(0,max(1,width-size+1),step)); ys=list(range(0,max(1,height-size+1),step))
    if xs[-1]!=max(0,width-size): xs.append(max(0,width-size))
    if ys[-1]!=max(0,height-size): ys.append(max(0,height-size))
    result.extend((x,y,min(width,x+size),min(height,y+size)) for y in ys for x in xs)
    return result


def yolo_boxes(history_row: dict, width: int, height: int) -> list[list[float]]:
    values=json.loads(history_row['after']); output=[]
    for box in values:
        x,y,w,h=(float(box[key]) for key in ('x','y','width','height'))
        output.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
    return output


def detect(model, processor, image: Image.Image, tile_size: int, overlap: float, threshold: float) -> list[dict]:
    width,height=image.size; raw=[]
    for left,top,right,bottom in tiles(width,height,tile_size,overlap):
        crop=image.crop((left,top,right,bottom)); inputs=processor(images=crop,text=[PROMPTS],return_tensors='pt')
        with torch.no_grad(): outputs=model(**inputs)
        result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=threshold,text_threshold=0.10,target_sizes=[crop.size[::-1]],text_labels=[PROMPTS])[0]
        for box,score,label in zip(result['boxes'],result['scores'],result['labels']):
            x1,y1,x2,y2=(float(value) for value in box.tolist())
            raw.append({'xyxy':[x1+left,y1+top,x2+left,y2+top],'confidence':float(score),'label':str(label),'tile':[left,top,right,bottom]})
    return nms(raw,0.35)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--images',type=Path,required=True); parser.add_argument('--history',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--summary',type=Path,required=True); parser.add_argument('--tile-size',type=int,default=1024); parser.add_argument('--overlap',type=float,default=.25); parser.add_argument('--threshold',type=float,default=.12); args=parser.parse_args()
    latest={}
    with args.history.open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle): latest[(row['split'],row['image'])]=row
    processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny',local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny',local_files_only=True).eval()
    fields=['split','image','gt_index','best_iou','matched_prediction','prediction_confidence','lid_like_prediction','detections','tiles']
    output=[]; totals={'gt':0,'matched_03':0,'matched_05':0,'predictions':0,'matched_predictions':0,'lid_like':0}; started=time.time()
    for index,((split,image),row) in enumerate(sorted(latest.items()),1):
        image_path=args.images/split/image
        image_data=Image.open(image_path).convert('RGB'); gt=yolo_boxes(row,*image_data.size)
        prediction=detect(model,processor,image_data,args.tile_size,args.overlap,args.threshold); totals['gt']+=len(gt); totals['predictions']+=len(prediction)
        pred_match=[max([iou(p['xyxy'],g) for g in gt] or [0]) for p in prediction]; totals['matched_predictions']+=sum(value>=.3 for value in pred_match)
        for gt_index,truth in enumerate(gt,1):
            best_index=-1; best_iou=0.0
            for candidate_index,candidate in enumerate(prediction):
                value=iou(truth,candidate['xyxy'])
                if value>best_iou: best_index,best_iou=candidate_index,value
            if best_iou>=.3: totals['matched_03']+=1
            if best_iou>=.5: totals['matched_05']+=1
            lid=False; confidence=''
            if best_index>=0:
                candidate=prediction[best_index]; confidence=f"{candidate['confidence']:.6f}"; area_truth=(truth[2]-truth[0])*(truth[3]-truth[1]); area_pred=(candidate['xyxy'][2]-candidate['xyxy'][0])*(candidate['xyxy'][3]-candidate['xyxy'][1]); lid=best_iou<.3 and area_pred<.45*area_truth and candidate['xyxy'][0]>=truth[0] and candidate['xyxy'][1]>=truth[1] and candidate['xyxy'][2]<=truth[2] and candidate['xyxy'][3]<=truth[3]
            if lid: totals['lid_like']+=1
            output.append({'split':split,'image':image,'gt_index':gt_index,'best_iou':f"{best_iou:.6f}",'matched_prediction':str(best_index),'prediction_confidence':confidence,'lid_like_prediction':str(lid).lower(),'detections':str(len(prediction)),'tiles':str(len(tiles(*image_data.size,args.tile_size,args.overlap)))})
        print(f'processed {index}/{len(latest)} {split}/{image} det={len(prediction)}',flush=True)
    with args.output.open('w',encoding='utf-8',newline='') as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader();writer.writerows(output)
    summary={'control_images':len(latest),'manual_full_frame_images':len(latest),'ground_truth_boxes':totals['gt'],'predictions_after_nms':totals['predictions'],'recall_iou_03':round(totals['matched_03']/totals['gt'],4) if totals['gt'] else 0,'recall_iou_05':round(totals['matched_05']/totals['gt'],4) if totals['gt'] else 0,'precision_iou_03':round(totals['matched_predictions']/totals['predictions'],4) if totals['predictions'] else 0,'lid_like_rate_of_gt':round(totals['lid_like']/totals['gt'],4) if totals['gt'] else 0,'tile_size':args.tile_size,'overlap':args.overlap,'threshold':args.threshold,'elapsed_seconds':round(time.time()-started,1),'note':'Only 18 unique manually confirmed full-frame images exist in geometry_changes.csv; this run cannot satisfy a 30-image manual calibration requirement.'}
    args.summary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
