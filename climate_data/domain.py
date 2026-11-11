"""清点温度场的面积与观察掩码。"""
from .io import unique,date,number,positive

def summarize(v):
    cells=unique(v['cells'],'id');areas={x['id']:positive(x['area']) for x in v['cells']};seen=set();bycell={x:0 for x in sorted(cells)};empty=[]
    for r in v['rows']:
        d=date(r['day'])
        if d in seen or set(r['values'])!=cells:raise ValueError('重复日或格引用不齐')
        seen.add(d);present=0
        for cell,x in r['values'].items():
            if x is not None:number(x);bycell[cell]+=1;present+=1
        if not present:empty.append(r['day'])
    for split in v['windows'].values():
        if date(split[0])>date(split[1]):raise ValueError('时间窗逆序')
    return {'cells':len(cells),'total_area':sum(areas.values()),'observations_per_cell':bycell,'empty_days':sorted(empty),'time_rows':len(seen)}
