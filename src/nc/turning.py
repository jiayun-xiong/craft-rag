"""Postprocessor for an explicitly specified teaching cell, using diameter X and metric G54."""
import math
import re
from pathlib import Path

import ezdxf
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from search import search
from src.retrieval.store import DEFAULT_INDEX


class Setup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    profile: Literal["Haas-XZ-demo-v1"]
    material: Literal["S45C"]
    hardness_hb: float = Field(ge=180,le=280)
    tool: Literal["MC6015 FP"]
    operation: Literal["负角外圆一般精加工"]
    stock_diameter_mm: float = Field(gt=0,le=80)
    free_length_mm: float = Field(gt=3,le=150)
    rpm: int = Field(gt=0,le=3500)
    feed_mm_rev: float = Field(gt=0,le=1)


def contour(path):
    if Path(path).stat().st_size > 8_000_000:
        raise ValueError("DXF 超过样件输入限制")
    doc = ezdxf.readfile(path)
    entities = list(doc.modelspace())
    if doc.units != 4 or len(entities)!=1 or entities[0].dxftype()!="LWPOLYLINE":
        raise ValueError("需要毫米 DXF，模型空间仅含一条 TURN_PROFILE 开放轮廓")
    line = entities[0]
    if (line.dxf.layer!="TURN_PROFILE" or line.closed or line.dxf.elevation or
            tuple(line.dxf.extrusion)!=(0,0,1) or line.dxf.const_width):
        raise ValueError("轮廓层、平面或宽度不符合约定")
    points=[]
    for z,r,w1,w2,bulge in line.get_points("xyseb"):
        if (bulge or w1 or w2 or not math.isfinite(z) or not math.isfinite(r)
                or r<=0 or round(z,3)!=z or round(2*r,3)!=2*r):
            raise ValueError("只支持直线、正半径及 0.001 mm 输出精度")
        points.append((float(z),float(2*r)))
    if not 2<=len(points)<=100 or points[0][0]!=0 or len(points)%2:
        raise ValueError("轮廓须从端面开始，逐段交替给出外圆及轴肩")
    sections=[]
    for i in range(0,len(points),2):
        (z,d),(end,other)=points[i:i+2]
        if d!=other or end>=z or (i and (z!=points[i-1][0] or d<=points[i-1][1])):
            raise ValueError("轮廓只支持由端面向卡盘方向直径递增的同轴外圆")
        sections.append((end,d))
    return sections


def conditions(hits):
    # All conditions must occur in fields of ONE table row.
    for h in hits:
        t=h["text"]
        if h["kind"]!="table_row" or not all(k in t for k in (
                "负角刀片","S45C","硬度: HB180─280","切削范围: 精加工","切削范围: 一般切削",
                "刀片材料: MC6015 |","断屑槽: FP |")):
            continue
        limits=[]
        for field in ("切削速度 vc (m/min)", "每转进给量 f (mm/rev)", "切削深度 ap"):
            m=re.search(re.escape(field)+r": (\d+(?:\.\d+)?)\s*[─～–-]\s*(\d+(?:\.\d+)?)",t)
            if not m: break
            limits.append(tuple(map(float,m.groups())))
        if len(limits)==3:
            return h,limits
    raise ValueError("缺少与材料、刀具、工况完全匹配的带单位参数行")


def plan(sections, setup, limits):
    speed,feed,depth=limits
    stock=setup.stock_diameter_mm
    if -sections[-1][0]+3>setup.free_length_mm:
        raise ValueError("轴向行程侵入预留的卡爪空间")
    if not feed[0]<=setup.feed_mm_rev<=feed[1]:
        raise ValueError("进给超出检索到的适用范围")
    passes=[]
    for end,d in reversed(sections):
        ap=round((stock-d)/2,6)
        if not depth[0]<=ap<=depth[1]:
            raise ValueError("单道精车余量超出适用切深，需要另编毛坯或粗加工工序")
        for diameter in (d,stock):
            if not speed[0]<=math.pi*diameter*setup.rpm/1000<=speed[1]:
                raise ValueError("固定转速对应的线速度超出选定工况")
        passes.append((d,end))
        stock=d
    return passes


def postprocess(passes, setup):
    clear=setup.stock_diameter_mm+4
    lines=["%","O0701","(HAAS XZ TEACHING CELL - REVIEW SETUP)",
           "G18 G21 G40 G80 G99","G54","T0101",f"G97 S{setup.rpm} M03","M08",
           f"G00 X{clear:.3f}","G00 Z2.000"]
    for d,end in passes:
        lines += [f"G00 X{d:.3f}",f"G01 Z{end:.3f} F{setup.feed_mm_rev:.4f}",
                  "G01 Z2.000",f"G00 X{clear:.3f}"]
    return "\n".join(lines+["M09","M05","M30","%",""])


def verify(code, sections, setup):
    """Replay the emitted axis words against a piecewise cylindrical stock model."""
    ends=[0]+[z for z,d in sections]
    stock=[setup.stock_diameter_mm]*len(sections)
    x,z=setup.stock_diameter_mm+4,2.0
    for line in code.splitlines():
        if not line.startswith(("G01 ","G00 ")): continue
        motion,axis,*rest=line.split()
        target=float(axis[1:])
        if axis[0]=="X":
            if z<=0: raise ValueError("在工件内部执行径向定位")
            x=target
        elif axis[0]=="Z":
            if target < -setup.free_length_mm+3: raise ValueError("轴向越界")
            for i,(a,b) in enumerate(zip(ends,ends[1:])):
                overlap=min(max(z,target),a)-max(min(z,target),b)
                if overlap<=0 or x>=stock[i]: continue
                if motion=="G00": raise ValueError("快速定位穿过毛坯")
                if overlap<a-b-1e-6: raise ValueError("走刀未覆盖完整轴段")
                stock[i]=x
            z=target
        else:
            raise ValueError("不支持的运动轴")
    if any(abs(actual-expected)>0.0005 for actual,(_,expected) in zip(stock,sections)):
        raise ValueError("回放所得外径不符图纸")
    return stock


def compile_job(path, setup, root=DEFAULT_INDEX):
    sections=contour(path)
    query=f"{setup.material} HB{setup.hardness_hb:g} HB180～280 负角外圆一般精加工 MC6015 FP 切削速度 每转进给 切削深度"
    h,limits=conditions(search(query,root,"rerank",5)["rerank"])
    passes=plan(sections,setup,limits)
    nc=postprocess(passes,setup)
    result=verify(nc,sections,setup)
    return nc,dict(profile=setup.profile,setup=setup.model_dump(),source=h["source"],page=h["page"],
                   table=h["table"],row=h["row"],evidence=h["text"],passes_xz_mm=passes,
                   replay_diameters_mm=result,validation="理想刀尖二维外径回放；需目标控制器仿真和装夹复核")
