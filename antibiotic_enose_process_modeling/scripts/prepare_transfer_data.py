#!/usr/bin/env python
"""Reproduce the Result-7 source/target ingestion from the original Excel files.

This script intentionally mirrors the archived raw-data ingestion logic used for the
current cross-antibiotic analysis, including Excel serial-time handling and the
±0.5-h sensor/offline alignment window.
"""
from pathlib import Path
import argparse, zipfile, xml.etree.ElementTree as ET, re, csv
import numpy as np

SOURCE_FILES=['KL01-1.xlsx','KL01-2.xlsx','TM-KL01-3.xlsx','0327-3748-1.xlsx','3748-1.xlsx','3756-1.xlsx']
TARGET_FILES={'Erythromycin':'红霉素电子嗅原始数据.xlsx','Cephalosporin':'头孢菌素电子嗅原始数据(1).xlsx','Lincomycin':'林可霉素电子嗅原始数据.xlsx'}
OFFLINE='发酵参数离线数据(1).xlsx'
SENS=[f'Channel_{i}' for i in range(1,17)]
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def col_idx(cell_ref):
    letters=re.match(r'([A-Z]+)',cell_ref).group(1);n=0
    for ch in letters:n=n*26+(ord(ch)-64)
    return n-1


def read_xlsx(path,sheet_name=None,sheet_index=0):
    """Minimal OOXML reader preserving numeric Excel serial values exactly."""
    with zipfile.ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si',NS):
                shared.append(''.join((t.text or '') for t in si.iter('{%s}t'%NS['m'])))
        wb=ET.fromstring(z.read('xl/workbook.xml'));sheets=wb.find('m:sheets',NS)
        sheet_meta=[(s.attrib['name'],s.attrib['{%s}id'%NS['r']]) for s in sheets]
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'));relmap={r.attrib['Id']:r.attrib['Target'] for r in rels}
        name,rid=sheet_meta[sheet_index] if sheet_name is None else next(x for x in sheet_meta if x[0]==sheet_name)
        target=relmap[rid]
        if target.startswith('/'):target=target.lstrip('/')
        elif not target.startswith('xl/'):target='xl/'+target
        root=ET.fromstring(z.read(target));rows=[];maxc=0
        for row in root.findall('.//m:sheetData/m:row',NS):
            vals={}
            for c in row.findall('m:c',NS):
                i=col_idx(c.attrib['r']);maxc=max(maxc,i+1);typ=c.attrib.get('t','n');v=c.find('m:v',NS)
                if typ=='inlineStr':
                    isel=c.find('m:is',NS);val=''.join((x.text or '') for x in isel.iter('{%s}t'%NS['m'])) if isel is not None else ''
                elif v is None:val=None
                else:
                    rawv=v.text
                    if typ=='s':val=shared[int(rawv)]
                    elif typ in ('str','e'):val=rawv
                    elif typ=='b':val=bool(int(rawv))
                    else:
                        try:
                            fv=float(rawv);val=int(fv) if fv.is_integer() else fv
                        except Exception:val=rawv
                vals[i]=val
            rows.append(vals)
        return [[r.get(i,None) for i in range(maxc)] for r in rows]


def write_csv(path,header,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)


def load_target_enose(raw_dir,product):
    vals=read_xlsx(raw_dir/TARGET_FILES[product]);rows=vals[1:];arr=[]
    if product=='Erythromycin':
        for r in rows:
            try:
                t=float(r[0]);x=np.array([float(v) for v in r[2:18]],float)
                if np.isfinite(t) and np.isfinite(x).all():arr.append((t,x))
            except Exception:pass
    else:
        tmp=[]
        for r in rows:
            try:
                ts=float(r[1]);x=np.array([float(v) for v in r[2:18]],float)
                if np.isfinite(ts) and np.isfinite(x).all():tmp.append((ts,x))
            except Exception:pass
        tmp.sort(key=lambda z:z[0]);t0=tmp[0][0];arr=[((ts-t0)*24.0,x) for ts,x in tmp]
    arr.sort(key=lambda z:z[0])
    return np.array([z[0] for z in arr]),np.vstack([z[1] for z in arr])


def align(t,X,tt,w=.5):
    out=[]
    for q in tt:
        m=np.abs(t-q)<=w
        if m.sum():out.append(np.median(X[m],axis=0))
        else:out.append(np.array([np.interp(q,t,X[:,j]) for j in range(16)]))
    return np.vstack(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',required=True);ap.add_argument('--output-dir',required=True)
    args=ap.parse_args();R=Path(args.raw_dir);O=Path(args.output_dir);O.mkdir(parents=True,exist_ok=True)
    # Gentamicin source: six real paired batches
    Xs=[];Ys=[];bn=[]
    for fn in SOURCE_FILES:
        vals=read_xlsx(R/fn)
        for r in vals[1:]:
            try:
                x=np.array([float(v) for v in r[:16]],float);y=np.array([float(v) for v in r[16:21]],float)
                if np.isfinite(x).all() and np.isfinite(y).all():Xs.append(x);Ys.append(y);bn.append(fn[:-5])
            except Exception:pass
    source_vars=['PMV','RS','TS','NH4','Titer']
    write_csv(O/'Gentamicin_REAL_source.csv',['Batch',*SENS,*source_vars],[[bn[i],*Xs[i],*Ys[i]] for i in range(len(Xs))])
    off_sheets={n:read_xlsx(R/OFFLINE,sheet_name=n) for n in ['红霉素离线数据','头孢菌素离线数据','林可霉素离线数据']}
    # Erythromycin
    rr=[]
    for r in off_sheets['红霉素离线数据'][1:]:
        try:rr.append([float(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[4])])
        except Exception:pass
    a=np.array(rr,float);t,X=load_target_enose(R,'Erythromycin');a=a[a[:,0]<=t.max()+1e-9];Xa=align(t,X,a[:,0])
    write_csv(O/'Erythromycin_REAL_aligned.csv',['Time_h',*SENS,'DCW','Titer','RS','NH4'],[[a[i,0],*Xa[i],*a[i,1:]] for i in range(len(a))])
    # Cephalosporin: TS/NH4 are absent in the real offline table and are not synthesized.
    rr=[]
    for r in off_sheets['头孢菌素离线数据'][1:]:
        try:rr.append([float(r[0]),float(r[1]),float(r[2]),float(r[3])])
        except Exception:pass
    a=np.array(rr,float);t,X=load_target_enose(R,'Cephalosporin');a=a[a[:,0]<=t.max()+1e-9];Xa=align(t,X,a[:,0])
    write_csv(O/'Cephalosporin_REAL_aligned.csv',['Time_h',*SENS,'DCW','Titer','RS'],[[a[i,0],*Xa[i],*a[i,1:]] for i in range(len(a))])
    # Lincomycin: three real offline replicates are averaged at common times because only one e-nose run exists.
    vals=off_sheets['林可霉素离线数据'];blocks=[];cur=[]
    for r in vals:
        if (r and r[0]=='Time (h)') or (not r) or r[0] is None:
            if cur:blocks.append(np.array(cur,float));cur=[]
            continue
        try:cur.append([float(r[0]),float(r[1]),float(r[2]),float(r[3]),float(r[5])])
        except Exception:pass
    if cur:blocks.append(np.array(cur,float))
    common=sorted(set.intersection(*[set(b[:,0]) for b in blocks]));avg=[]
    for tt in common:
        q=np.array([b[b[:,0]==tt][0,1:] for b in blocks]);avg.append([tt,*q.mean(axis=0)])
    a=np.array(avg,float);t,X=load_target_enose(R,'Lincomycin');a=a[a[:,0]<=t.max()+1e-9];Xa=align(t,X,a[:,0])
    write_csv(O/'Lincomycin_REAL_aligned.csv',['Time_h',*SENS,'PMV','Titer','RS','NH4'],[[a[i,0],*Xa[i],*a[i,1:]] for i in range(len(a))])
    print('Prepared transfer-learning tables in',O)
if __name__=='__main__':main()
