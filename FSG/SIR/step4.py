#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将三个图合并为统一编号的一个图，并新增“测试覆盖”：
- 方法调用图（method_graph.txt：methodnodes/edge）
- 方法+可疑代码行图（codeline_flow.txt：混合 method/codeline + edge）
- 变量数据流图目录（var_dir/N.txt：node/edge；N= codeline_flow 的本地 codeline 编号）
- 测试覆盖文件（--testfile）：
    第1行：测试名
    第2行：失败报告
    第3行起：覆盖行，形如 org.apache.commons.lang3.time$FormatCache#FormatCache():171

输出：
nodes:
<gid> method: {content: ...}
<gid> codeline: {content: ..., type: ..., sus: ...}
<gid> var: {content: ..., type: ...}
<gid> [test] {content: 测试名, report: 报告}
edge:
u->v   # 统一后的所有边（方法边、方法内顺序、变量内部、codeline→变量头、测试→codeline）
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

WS_RE = re.compile(r"\s+")
def norm(s: str) -> str:
    return WS_RE.sub(" ", s.strip())

# ---------- 解析 方法调用图 ----------
def parse_method_graph(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    mode = None
    id_to_method: Dict[int, str] = {}
    method_edges_local: List[Tuple[int,int]] = []
    for ln in lines:
        if ln.lower().startswith("methodnodes"):
            mode = "nodes"; continue
        if ln == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+method:\s*\{content:\s*(.+?)\s*\}\s*$", ln)
            if m:
                idx = int(m.group(1))
                content = norm(m.group(2))
                id_to_method[idx] = content
        elif mode == "edges":
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                method_edges_local.append((int(m.group(1)), int(m.group(2))))
    method_edges_by_name: List[Tuple[str,str]] = []
    for u,v in method_edges_local:
        if u in id_to_method and v in id_to_method:
            method_edges_by_name.append((id_to_method[u], id_to_method[v]))
    return list(id_to_method.values()), method_edges_by_name

# ---------- 解析 方法+代码行图 ----------
def parse_codeline_flow(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    nodes_seq: List[Tuple[str, Dict[str,str]]] = []
    edges_local: List[Tuple[int,int]] = []
    codeline_local_ids: List[int] = []
    mode = "nodes"
    for ln in lines:
        if ln == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s+(method|codeline):\s*\{(.+)\}\s*$", ln)
            if not m:
                continue
            idx = int(m.group(1))
            kind = m.group(2)
            body = m.group(3)
            payload: Dict[str,str] = {}
            for k,v in re.findall(r"(\w+)\s*:\s*([^,}]+)", body):
                payload[k] = norm(v)
            nodes_seq.append((kind, payload))
            if kind == "codeline":
                codeline_local_ids.append(idx)
        else:
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                edges_local.append((int(m.group(1)), int(m.group(2))))
    return nodes_seq, edges_local, codeline_local_ids

# ---------- 解析 变量图（单文件） ----------
def parse_var_file(text: str):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    mode = None
    nodes: List[Tuple[str,str]] = []
    edges: List[Tuple[int,int]] = []
    for ln in lines:
        low = ln.lower()
        if low == "node:":
            mode = "nodes"; continue
        if low == "edge:":
            mode = "edges"; continue
        if mode == "nodes":
            m = re.match(r"(\d+)\s*\{\s*content:\s*([^,}]+)\s*,\s*type:\s*([^}]+)\}\s*$", ln)
            if m:
                name = norm(m.group(2))
                vtype = norm(m.group(3))
                nodes.append((name, vtype))
        elif mode == "edges":
            m = re.match(r"(\d+)\s*->\s*(\d+)$", ln)
            if m:
                edges.append((int(m.group(1)), int(m.group(2))))
    return nodes, edges

# ---------- 覆盖行解析（测试文件） ----------
COV_RE = re.compile(
    r"""
    ^
    (?P<pkg>[A-Za-z_][\w\.]*)
    \$
    (?P<classfull>[A-Za-z_]\w*(?:\$(?:[A-Za-z_]\w*|\d+))*)   # 允许内部类形式（按原格式保留）
    \#
    (?P<meth>[A-Za-z_]\w*|<init>|<clinit>)
    \(\)
    :
    (?P<line>\d+)
    $
    """, re.VERBOSE
)

def parse_testfile(p: Path) -> Tuple[str, str, List[Tuple[str,int]]]:
    """
    返回：测试名，报告，[ (方法名(仅函数名), 行号) ... ]
    这里的方法名仅取 '#' 后的标识符（用于与方法全名的 #后缀匹配）
    """
    raw = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    name = norm(raw[0]) if len(raw) >= 1 else ""
    report = norm(raw[1]) if len(raw) >= 2 else ""
    cov: List[Tuple[str,int]] = []
    for ln in raw[2:]:
        s = ln.strip()
        if not s:
            continue
        m = COV_RE.match(s)
        if m:
            fn = m.group("meth")
            line = int(m.group("line"))
            cov.append((fn, line))
    return name, report, cov

# ---------- 全局编号 ----------
class GlobalIndex:
    def __init__(self):
        self.nodes_out: List[str] = []
        self.key_to_gid: Dict[Tuple, int] = {}

    def add_method(self, content: str) -> int:
        key = ("method", content)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} method: {{content: {content}}}")
        return gid

    def add_codeline(self, content: str, linetype: str, sus: str) -> int:
        key = ("codeline", content, linetype, sus)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} codeline: {{content: {content}, type: {linetype}, sus: {sus}}}")
        return gid

    def add_var(self, name: str, vtype: str) -> int:
        key = ("var", name, vtype)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} var: {{content: {name}, type: {vtype}}}")
        return gid

    def add_test(self, content: str, report: str) -> int:
        key = ("test", content, report)
        if key in self.key_to_gid:
            return self.key_to_gid[key]
        gid = len(self.nodes_out) + 1
        self.key_to_gid[key] = gid
        self.nodes_out.append(f"{gid} [test] {{content: {content}, report: {report}}}")
        return gid

def main():
    ap = argparse.ArgumentParser(description="合并三图 + 测试覆盖为统一编号图")
    ap.add_argument("--method_graph", type=Path, required=True)
    ap.add_argument("--codeline_flow", type=Path, required=True)
    ap.add_argument("--var_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--testfile", type=Path, required=False, help="测试名.txt：第1行测试名，第2行报告，后续覆盖行")
    args = ap.parse_args()

    gi = GlobalIndex()
    edges_out: List[str] = []

    # ========== 1) 先读 codeline_flow，建立节点 ==========
    cl_text = args.codeline_flow.read_text(encoding="utf-8", errors="ignore")
    cl_nodes, cl_edges_local, cl_codeline_ids = parse_codeline_flow(cl_text)

    local_id_to_gid: Dict[int, int] = {}
    method_name_to_gid: Dict[str, int] = {}

    # 还要记录：method_full -> 该方法下的所有 codeline(全局)；以及 短函数名 -> 多个 method_full
    method_to_codeline_gids: Dict[str, List[int]] = {}
    shortname_to_methods: Dict[str, List[str]] = {}

    current_method_full: Optional[str] = None
    for local_idx, (kind, payload) in enumerate(cl_nodes, start=1):
        if kind == "method":
            mname = payload.get("content","")
            gid = gi.add_method(mname)
            local_id_to_gid[local_idx] = gid
            method_name_to_gid.setdefault(mname, gid)
            current_method_full = mname
            # 维护短函数名映射
            if "#" in mname:
                short = mname.split("#",1)[1]
                shortname_to_methods.setdefault(short, []).append(mname)
            method_to_codeline_gids.setdefault(mname, [])
        else:
            gid = gi.add_codeline(payload.get("content",""), payload.get("type",""), payload.get("sus",""))
            local_id_to_gid[local_idx] = gid
            if current_method_full:
                method_to_codeline_gids.setdefault(current_method_full, []).append(gid)

    # 方法内顺序边 / 方法->首条行
    for u,v in cl_edges_local:
        ug = local_id_to_gid.get(u)
        vg = local_id_to_gid.get(v)
        if ug and vg:
            edges_out.append(f"{ug}->{vg}")

    # ========== 2) 读方法调用图，补齐方法 & 方法边 ==========
    mg_text = args.method_graph.read_text(encoding="utf-8", errors="ignore")
    mg_methods, mg_edges_by_name = parse_method_graph(mg_text)

    for mname in mg_methods:
        if mname not in method_name_to_gid:
            gid = gi.add_method(mname)
            method_name_to_gid[mname] = gid
            if "#" in mname:
                short = mname.split("#",1)[1]
                shortname_to_methods.setdefault(short, []).append(mname)
            method_to_codeline_gids.setdefault(mname, [])

    for a_name, b_name in mg_edges_by_name:
        ag = method_name_to_gid.get(a_name)
        bg = method_name_to_gid.get(b_name)
        if ag and bg:
            edges_out.append(f"{ag}->{bg}")

    # ========== 3) 合入变量图 ==========
    if not args.var_dir.is_dir():
        raise ValueError(f"--var_dir 不是目录：{args.var_dir}")

    for f in sorted(args.var_dir.glob("*.txt"), key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else 10**12, p.stem)):
        stem = f.stem
        if not stem.isdigit():
            continue
        local_codeline_id = int(stem)
        codeline_gid = local_id_to_gid.get(local_codeline_id)
        if not codeline_gid:
            continue

        vtext = f.read_text(encoding="utf-8", errors="ignore")
        v_nodes, v_edges = parse_var_file(vtext)

        v_local_to_gid: Dict[int,int] = {}
        for i,(name,vtype) in enumerate(v_nodes, start=1):
            gid = gi.add_var(name, vtype)
            v_local_to_gid[i] = gid

        indeg: Dict[int,int] = {i:0 for i in v_local_to_gid.keys()}
        for u,v in v_edges:
            ug = v_local_to_gid.get(u)
            vg = v_local_to_gid.get(v)
            if ug and vg:
                edges_out.append(f"{ug}->{vg}")
                indeg[v] = indeg.get(v,0) + 1

        for i,deg in indeg.items():
            if deg == 0 and i in v_local_to_gid:
                edges_out.append(f"{codeline_gid}->{v_local_to_gid[i]}")

    # ========== 4) 测试覆盖（可选） ==========
    if args.testfile and args.testfile.exists():
        t_name, t_report, cov = parse_testfile(args.testfile)
        test_gid = gi.add_test(t_name, t_report)

        # 将测试节点连到覆盖行所属方法里的 codeline 节点：
        # 以“短函数名”匹配（覆盖串中 meth 与 我们的方法全名 content 的 #后缀）
        for (fn_short, _line) in cov:
            methods = shortname_to_methods.get(fn_short, [])
            for mfull in methods:
                for cl_gid in method_to_codeline_gids.get(mfull, []):
                    edges_out.append(f"{test_gid}->{cl_gid}")

    # ========== 5) 输出 ==========
    out_lines: List[str] = []
    out_lines.append("nodes:")
    out_lines.extend(gi.nodes_out)
    out_lines.append("edge:")
    out_lines.extend(edges_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"合并后的统一图已写入：{args.out}")

if __name__ == "__main__":
    main()
