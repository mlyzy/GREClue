#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-Sitter(C) + 怀疑列表（locs）：
- 解析 locs（org.xxx$File#Func():line;sus）
- 全项目构建“函数 -> 直接被调函数”图
- 以可疑方法为种子做向外 BFS（多跳），仅输出可达子图
- 只输出两段：
  methodnodes:
    N method: {content: <相对路径去扩展名>#<函数名>}
  edge:
    u->v

依赖：
  pip install tree_sitter tree_sitter_languages
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, deque
from tree_sitter import Parser

# ----------------- 解析 locs -----------------
# 例如：org.apache.commons.lang3.time$FormatCache#FormatCache():41;1.0
LOC_RE = re.compile(
    r"""
    ^
    (?P<pkg>[A-Za-z_][\w\.]*)      # org.apache.commons.lang3.time
    \$
    (?P<filebase>[A-Za-z_]\w*)     # FormatCache
    \#
    (?P<func>[A-Za-z_]\w*)         # FormatCache（C 函数名）
    \(\)
    :
    (?P<line>\d+)
    ;
    (?P<sus>\d+(?:\.\d+)?)
    $
    """,
    re.VERBOSE
)

class Susp:
    def __init__(self, raw: str, pkg: str, filebase: str, func: str, line: int, sus: float):
        self.raw = raw
        self.pkg = pkg
        self.filebase = filebase
        self.func = func
        self.line = line
        self.sus = sus

    @property
    def rel_hint(self) -> Path:
        # 期望目录 = pkg 替换 '.' 为目录；文件 = filebase.c
        return Path(*self.pkg.split(".")) / f"{self.filebase}.c"

def parse_loc_line(s: str) -> Susp:
    m = LOC_RE.match(s.strip())
    if not m:
        raise ValueError(f"非法位置串：{s}")
    return Susp(
        raw=s.strip(),
        pkg=m.group("pkg"),
        filebase=m.group("filebase"),
        func=m.group("func"),
        line=int(m.group("line")),
        sus=float(m.group("sus")),
    )

# ----------------- Tree-Sitter 函数/调用抽取 -----------------
def load_c_language():
    try:
        from tree_sitter_languages import get_language
        return get_language("c")
    except Exception:
        raise RuntimeError("请先安装依赖：pip install tree_sitter tree_sitter_languages")

def node_text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()

def walk(node):
    yield node
    for i in range(node.named_child_count):
        yield from walk(node.named_child(i))

def extract_functions_and_calls_c(src_path: Path, parser: Parser) -> Dict[str, Tuple[int,int,Set[str]]]:
    """
    返回：func_name -> (start_line, end_line, direct_callees)
    - 仅抽取 call_expression 的标识符调用（identifier(...)）
    """
    source = src_path.read_bytes()
    tree = parser.parse(source)
    root = tree.root_node
    out: Dict[str, Tuple[int,int,Set[str]]] = {}

    for defn in walk(root):
        if defn.type != "function_definition":
            continue

        # 函数名（在 function_declarator/declarator 的 identifier）
        func_name = None
        decl = None
        for i in range(defn.named_child_count):
            ch = defn.named_child(i)
            if ch.type in ("function_declarator", "declarator"):
                decl = ch
                break
        if decl is not None:
            stack = [decl]
            while stack:
                n = stack.pop()
                if n.type == "identifier":
                    func_name = node_text(source, n)
                    break
                for j in range(n.named_child_count):
                    stack.append(n.named_child(j))
        if not func_name:
            continue

        # 函数体范围
        body = None
        for i in range(defn.named_child_count):
            ch = defn.named_child(i)
            if ch.type == "compound_statement":
                body = ch
                break
        if body is None:
            continue

        start_line = body.start_point[0] + 1
        end_line   = body.end_point[0] + 1

        # 抓直接调用
        callees: Set[str] = set()
        for inner in walk(body):
            if inner.type == "call_expression" and inner.named_child_count > 0:
                fn = inner.named_child(0)
                if fn.type == "identifier":
                    callee = node_text(source, fn)
                    if callee:
                        callees.add(callee)

        out[func_name] = (start_line, end_line, callees)

    return out

# ----------------- 项目索引与图构建 -----------------
def read_lines(p: Path) -> List[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore").splitlines()

def find_c_file(project_root: Path, rel_hint: Path, filebase: str) -> Path | None:
    cand = [
        project_root / rel_hint,
        project_root / "src" / rel_hint,
        project_root / "source" / rel_hint,
        project_root / "lib" / rel_hint,
        project_root / "src" / "c" / rel_hint,
    ]
    for p in cand:
        if p.is_file():
            return p
    for p in project_root.rglob(f"{filebase}.c"):
        return p
    return None

def index_project(project_root: Path) -> Tuple[Dict[str, Dict[str, Tuple[int,int,Set[str]]]], Dict[str,str]]:
    """
    扫描 *.c，返回：
      file_to_funcs: relpath -> { func: (start, end, callees) }
      func_to_fullname: func -> "<rel_no_ext>#func"（首次定义）
    """
    LANG_C = load_c_language()
    parser = Parser()
    parser.set_language(LANG_C)

    file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]] = {}
    func_to_fullname: Dict[str, str] = {}

    for cfile in sorted(project_root.rglob("*.c")):
        try:
            fmap = extract_functions_and_calls_c(cfile, parser)
        except Exception:
            continue
        if not fmap:
            continue
        rel = str(cfile.relative_to(project_root))
        file_to_funcs[rel] = fmap
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn in fmap.keys():
            func_to_fullname.setdefault(fn, f"{base_no_ext}#{fn}")
    return file_to_funcs, func_to_fullname

def build_direct_graph(file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]]) -> Dict[str, Set[str]]:
    g: Dict[str, Set[str]] = defaultdict(set)
    for _rel, fmap in file_to_funcs.items():
        for fn, (_s,_e,callees) in fmap.items():
            g[fn].update(callees)
    return g

# ----------------- 由怀疑列表确定种子方法 -----------------
def locate_owner_func_for_line(cfile: Path, target_line: int, fmap: Dict[str, Tuple[int,int,Set[str]]]) -> str | None:
    for fn, (s, e, _c) in fmap.items():
        if s <= target_line <= e:
            return fn
    return None

def seeds_from_locs(project_root: Path,
                    file_to_funcs: Dict[str, Dict[str, Tuple[int,int,Set[str]]]],
                    suspicious: List[Susp]) -> Set[str]:
    seeds: Set[str] = set()
    # 建立 rel -> fmap 的索引
    for sp in suspicious:
        cpath = find_c_file(project_root, sp.rel_hint, sp.filebase)
        if not cpath:
            # 找不到文件，退回到显式函数名
            seeds.add(sp.func)
            continue
        rel = str(cpath.relative_to(project_root))
        fmap = file_to_funcs.get(rel)
        if not fmap:
            # 该文件可能语法异常而被跳过，退回函数名
            seeds.add(sp.func)
            continue
        owner = locate_owner_func_for_line(cpath, sp.line, fmap)
        seeds.add(owner if owner else sp.func)
    return seeds

# ----------------- 只输出可达子图（从种子出发） -----------------
def reachable_subgraph_from(seeds: Set[str],
                            g: Dict[str, Set[str]]) -> Set[str]:
    """从种子做 BFS，得到所有可达函数名（含种子本身）"""
    reach: Set[str] = set()
    q = deque(seeds)
    reach.update(seeds)
    while q:
        u = q.popleft()
        for v in g.get(u, ()):
            if v not in reach:
                reach.add(v)
                q.append(v)
    return reach

# ----------------- 主程序 -----------------
def main():
    ap = argparse.ArgumentParser(description="Tree-Sitter(C) 调用图（带怀疑列表，输出 methodnodes/edge）")
    ap.add_argument("--project", required=True, type=Path, help="C 项目根目录")
    ap.add_argument("--locs", required=True, type=Path, help="怀疑位置串文件，每行 org.xxx$File#Func():line;sus")
    ap.add_argument("--out", required=True, type=Path, help="输出 txt")
    args = ap.parse_args()

    project_root = args.project.resolve()
    loc_lines = [ln.strip() for ln in args.locs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    suspicious = [parse_loc_line(s) for s in loc_lines]

    # 1) 项目索引 & 直接调用图
    file_to_funcs, func_to_fullname = index_project(project_root)
    g = build_direct_graph(file_to_funcs)

    # 2) 由怀疑列表得到“种子函数”
    seeds = seeds_from_locs(project_root, file_to_funcs, suspicious)

    # 3) 只保留从种子出发可达的函数（子图）
    keep_funcs = reachable_subgraph_from(seeds, g)

    # 4) 构建 fullname，并统一编号
    fullname_to_id: Dict[str, int] = {}
    nodes_out: List[str] = []
    next_id = 1

    # 我们需要稳定顺序：按文件、函数名排序，但只收录 keep_funcs 中存在且在项目中有定义的位置
    # 建立 “函数名 -> 其定义文件相对路径（首次）” 的反向帮助（已在 func_to_fullname 中）
    for rel in sorted(file_to_funcs.keys()):
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn in sorted(file_to_funcs[rel].keys()):
            if fn not in keep_funcs:
                continue
            fullname = f"{base_no_ext}#{fn}"
            if fullname not in fullname_to_id:
                fullname_to_id[fullname] = next_id
                nodes_out.append(f"{next_id} method: {{content: {fullname}}}")
                next_id += 1

    # 5) 边：只输出子图中（两端函数都在 keep_funcs）的直接调用边
    edges_out: List[str] = []

    # 帮助把“函数名”映射到 fullname 与 id
    def fn_to_full(fn: str) -> str | None:
        # 若该函数在 keep 中且在项目中出现过
        return func_to_fullname.get(fn)

    def full_to_id(full: str) -> int | None:
        return fullname_to_id.get(full)

    for rel, fmap in file_to_funcs.items():
        base_no_ext = str(Path(rel).with_suffix(""))
        for fn, (_s,_e,callees) in fmap.items():
            if fn not in keep_funcs:
                continue
            src_full = f"{base_no_ext}#{fn}"
            src_id = full_to_id(src_full)
            if not src_id:
                continue
            for cal in sorted(callees):
                if cal not in keep_funcs:
                    continue
                dst_full = fn_to_full(cal)
                if not dst_full:
                    continue
                dst_id = full_to_id(dst_full)
                if dst_id:
                    edges_out.append(f"{src_id}->{dst_id}")

    # 6) 写出
    out_lines: List[str] = []
    out_lines.append("methodnodes:")
    out_lines.extend(nodes_out)
    out_lines.append("edge:")
    out_lines.extend(edges_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"已写入：{args.out}")

if __name__ == "__main__":
    main()
