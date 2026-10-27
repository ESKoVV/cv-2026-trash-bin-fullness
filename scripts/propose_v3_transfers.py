#!/usr/bin/env python3
"""Read-only homography proposals from manually confirmed v3 frames.

Requires OpenCV. Writes proposals only; it never edits labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.webp'}

def paths(root: Path):
    return [(split,p) for split in ('train','val','test') for p in sorted((root/'images'/split).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

def image_data(path: Path, orb):
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'cannot read {path}')
    small=cv2.resize(image,(96,96));hist=cv2.normalize(cv2.calcHist([small],[0,1,2],None,[8,8,8],[0,256]*3),None).flatten()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if max(gray.shape)>900:
        scale=900/max(gray.shape);gray=cv2.resize(gray,(round(gray.shape[1]*scale),round(gray.shape[0]*scale)))
    keypoints,descriptors=orb.detectAndCompute(gray,None)
    return image,gray,keypoints,descriptors,hist

def box_corners(box, width, height):
    x,y,w,h=(float(box[key]) for key in ('x','y','width','height'))
    return np.float32([[(x-w/2)*width,(y-h/2)*height],[(x+w/2)*width,(y-h/2)*height],[(x+w/2)*width,(y+h/2)*height],[(x-w/2)*width,(y+h/2)*height]]).reshape(-1,1,2)

def project(box, transform, src_size, dst_size):
    points=cv2.perspectiveTransform(box_corners(box,*src_size),transform).reshape(-1,2)
    left,top=points.min(axis=0);right,bottom=points.max(axis=0);width,height=dst_size
    cx=(left+right)/2/width;cy=(top+bottom)/2/height;bw=(right-left)/width;bh=(bottom-top)/height
    if not (0<cx<1 and 0<cy<1 and .005<bw<.95 and .005<bh<.95 and 0<=cx-bw/2 and cx+bw/2<=1 and 0<=cy-bh/2 and cy+bh/2<=1):return None
    return {'class_id':int(box['class_id']),'x':round(float(cx),6),'y':round(float(cy),6),'width':round(float(bw),6),'height':round(float(bh),6)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--history',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--top',type=int,default=35);parser.add_argument('--include-manual-targets',action='store_true',help='calibration only: permit manually confirmed frames as targets');args=parser.parse_args()
    history=list(csv.DictReader(args.history.open(encoding='utf-8'))); latest={}
    for row in history: latest[(row['split'],row['image'])]=row
    orb=cv2.ORB_create(nfeatures=1200,fastThreshold=12); all_paths=paths(args.data); cache={}
    for split,path in all_paths: cache[(split,path.name)]=image_data(path,orb)
    references=[]
    for key,row in latest.items():
        if key not in cache: continue
        references.append((key,row,json.loads(row['after']),cache[key]))
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    rows=[]
    for ref_key,ref_row,ref_boxes,ref_data in references:
        _,ref_gray,ref_kp,ref_desc,ref_hist=ref_data
        if ref_desc is None or len(ref_kp)<20: continue
        ranked=[]
        for key,data in cache.items():
            if key == ref_key or (key in latest and not args.include_manual_targets): continue
            corr=float(cv2.compareHist(ref_hist.astype('float32'),data[4].astype('float32'),cv2.HISTCMP_CORREL))
            ranked.append((corr,key,data))
        for histogram_similarity,key,data in sorted(ranked,reverse=True,key=lambda x:x[0])[:args.top]:
            _,candidate_gray,candidate_kp,candidate_desc,_=data
            if candidate_desc is None or len(candidate_kp)<20:continue
            pairs=matcher.knnMatch(ref_desc,candidate_desc,k=2)
            good=[first for first,second in pairs if first.distance < .72*second.distance]
            if len(good)<12:continue
            src=np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2);dst=np.float32([candidate_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            transform,mask=cv2.findHomography(src,dst,cv2.RANSAC,4.0)
            if transform is None or mask is None:continue
            inliers=int(mask.ravel().sum()); ratio=inliers/len(good)
            score=max(0.0,min(1.0,histogram_similarity))*ratio*min(1.0,inliers/70)
            level='high' if inliers>=32 and ratio>=.55 and histogram_similarity>=.55 and score>=.42 else 'medium' if inliers>=16 and ratio>=.40 else 'low'
            for index,box in enumerate(ref_boxes,1):
                proposal=project(box,transform,(ref_gray.shape[1],ref_gray.shape[0]),(candidate_gray.shape[1],candidate_gray.shape[0]))
                if proposal is None:continue
                rows.append({'split':key[0],'image':key[1],'reference_split':ref_key[0],'reference_image':ref_key[1],'reference_event':ref_row['timestamp'],'reference_box_index':index,'class_id':proposal['class_id'],'x':proposal['x'],'y':proposal['y'],'width':proposal['width'],'height':proposal['height'],'confidence':round(score,4),'level':level,'method':'visual_feature_homography_transfer','histogram_similarity':round(histogram_similarity,4),'feature_matches':len(good),'homography_inliers':inliers,'inlier_ratio':round(ratio,4)})
    fields=['split','image','reference_split','reference_image','reference_event','reference_box_index','class_id','x','y','width','height','confidence','level','method','histogram_similarity','feature_matches','homography_inliers','inlier_ratio']
    with args.output.open('w',encoding='utf-8',newline='') as out:writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'references':len(references),'images':len(all_paths),'proposals':len(rows),'high':sum(x['level']=='high' for x in rows),'medium':sum(x['level']=='medium' for x in rows),'low':sum(x['level']=='low' for x in rows)}))

if __name__=='__main__':main()
