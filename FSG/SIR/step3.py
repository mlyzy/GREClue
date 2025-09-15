#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取某目录下的 N.txt（每个文件仅一行 C/类C 代码），
对行内变量做启发式数据流分析，并将结果写到另一个目录的 N.txt：

文件内容格式：
node:
1 {content: 变量名, type: 变量类型}
2 {content: 变量名, type: 变量类型}
edge:
1->2
...

规则（单行启发式）：
1) 赋值 a = expr       : expr 中每个变量 -> a
2) 复合赋值 a += expr  : expr 中每个变量 -> a（忽略自环）
3) 函数赋值 x = f(a,b) : a->x, b->x
4) 三目   x = c ? a:b  : c->x, a->x, b->x
5) 无赋值（return/调用）：按出现顺序建立有向链 v1->v2->...->vk

声明识别（给出类型）：
- 形如 "const unsigned long *x, y = 0;" / "struct S* p;"
- 仅对**声明的变量**标注检测到的类型，其他一律 unknown
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Set, Dict, Tuple

IDENT_RE = re.compile(r"\b[_A-Za-z]\w*\b")
# 粗略的 C 关键字与常见类型名
C_KEYWORDS: Set[str] = {
    "auto","break","case","char","const","continue","default","do","double","else",
    "enum","extern","float","for","goto","if","inline","int","long","register",
    "restrict","return","short","signed","sizeof","static","struct","switch",
    "typedef","union","unsigned","void","volatile","while","_Bool","_Complex","_Atomic"
}
COMMON_TYPES: Set[str] = {
    "size_t","ssize_t","uint8_t","uint16_t","uint32_t","uint64_t",
    "int8_t","int16_t","int32_t","int64_t","bool","BOOL"
}

ASSIGN_OPS = [
    "<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
    "="  # 放最后防止被上面的包含
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def split_first_assignment(line: str) -> Tuple[str, str, str] | None:
    """
    返回 (lhs, op, rhs)；若非赋值，返回 None
    避免将 ==, !=, <=, >= 当作赋值
    """
    i = 0
    while i < len(line):
        if line[i] == '=':
            # 检查是否比较运算
            prev = line[i-1] if i-1 >= 0 else ''
            nxt = line[i+1] if i+1 < len(line) else ''
            if prev in ['<','>','!','=',] or nxt == '=':
                i += 1
                continue
            # 找到最右侧的复合赋值匹配
        i += 1
    # 用预定义顺序查找第一个出现的赋值运算符
    # 为了避免被 '==' 等干扰，先替换比较为占位
    tmp = line.replace("==", "  ").replace("!=", "  ").replace("<=", "  ").replace(">=", "  ")
    pos = []
    for op in ASSIGN_OPS:
        j = tmp.find(op)
        if j != -1:
            pos.append((j, op))
    if not pos:
        return None
    j, op = sorted(pos, key=lambda x: x[0])[0]
    lhs = line[:j].strip()
    rhs = line[j+len(op):].strip().rstrip(';')
    return lhs, op, rhs

def extract_identifiers(s: str) -> List[str]:
    """提取串中的标识符，保持出现顺序；过滤关键字与常见类型"""
    ids = []
    for m in IDENT_RE.finditer(s):
        w = m.group(0)
        if w in C_KEYWORDS or w in COMMON_TYPES:
            continue
        ids.append(w)
    return ids

def extract_declared_vars_and_type(line: str) -> Dict[str, str]:
    """
    识别简单声明，返回 {变量名: 类型串}
    支持：
      <type> <var>(=...)? (, <var>(=...)?)* ;
    type 允许含 const/unsigned/signed/struct/指针星号等
    """
    out: Dict[str, str] = {}
    s = line.strip().rstrip(';')
    # 排除函数声明/定义（后随 '(' 紧跟标识符的）
    if re.search(r"\)\s*\{", s) or re.search(r"\w+\s*\([^)]*\)\s*$", s):
        return out

    m = re.match(
        r"^(?P<type>(?:const\s+|unsigned\s+|signed\s+)*"
        r"(?:struct\s+\w+|\w+)"
        r"(?:\s*\*+)*)\s+(?P<rest>.+)$",
        s
    )
    if not m:
        return out
    typestr = norm(m.group("type"))
    rest = m.group("rest")
    # 拆逗号
    parts = [p.strip() for p in rest.split(",")]
    for p in parts:
        # 变量名可能后跟 "[]", "= expr" 等
        # 取第一个标识符作为变量名
        m2 = IDENT_RE.search(p)
        if not m2:
            continue
        name = m2.group(0)
        if name in C_KEYWORDS:
            continue
        out[name] = typestr
    return out

def parse_ternary(rhs: str) -> Tuple[str, str, str] | None:
    # 粗略匹配 cond ? a : b
    m = re.search(r"(.+?)\?(.+?):(.+)", rhs)
    if not m:
        return None
    cond = m.group(1)
    a = m.group(2)
    b = m.group(3)
    return cond, a, b

def analyze_line(line: str) -> Tuple[List[Tuple[str,str]], List[Tuple[int,int]]]:
    """
    分析单行：
      返回 (nodes, edges)
      nodes: [(var, type)], 顺序依照首次出现
      edges: [(i,j)] 使用 nodes 的 1-based 索引
    """
    line = norm(line)
    # 1) 识别声明并记录类型
    var_types = extract_declared_vars_and_type(line)  # 可能为空

    # 2) 节点收集 & 建立可重复到唯一索引的顺序
    vars_order: List[str] = []   # 按首次出现
    def add_var(v: str):
        if v and v not in vars_order:
            vars_order.append(v)

    # 3) 赋值/复合赋值
    edges: List[Tuple[int,int]] = []

    asg = split_first_assignment(line)
    if asg:
        lhs, op, rhs = asg
        # LHS 可能有指针/字段，取基础标识符
        lhs_ids = extract_identifiers(lhs)
        # 常见情况下取第一个作为目标变量
        lhs_vars = [lhs_ids[0]] if lhs_ids else []
        for v in lhs_vars:
            add_var(v)

        # 处理三目
        tern = parse_ternary(rhs)
        if tern:
            cond, a, b = tern
            rhs_vars = extract_identifiers(cond) + extract_identifiers(a) + extract_identifiers(b)
        else:
            rhs_vars = extract_identifiers(rhs)

        for v in rhs_vars:
            add_var(v)

        # 建边：rhs 每个变量 -> lhs 变量
        for r in rhs_vars:
            for lv in lhs_vars:
                if r != lv:
                    edges.append((vars_order.index(r)+1, vars_order.index(lv)+1))

        # 复合赋值（如 x += y），也可把旧 x 视为参与者（已在 lhs_vars 中，且上面已过滤自环）
        # 不额外处理自环

    else:
        # 4) 无赋值：提取全部变量并按出现顺序连接成有向链
        ids = extract_identifiers(line)
        for v in ids:
            add_var(v)
        for i in range(len(vars_order)-1):
            edges.append((i+1, i+2))

    # 5) 组装节点（带类型；未知为 unknown）
    nodes: List[Tuple[str,str]] = []
    for v in vars_order:
        vtype = var_types.get(v, "unknown")
        nodes.append((v, vtype))

    return nodes, edges

def write_one(out_path: Path, nodes: List[Tuple[str,str]], edges: List[Tuple[int,int]]):
    lines: List[str] = []
    lines.append("node:")
    for i, (v, t) in enumerate(nodes, start=1):
        lines.append(f"{i} {{content: {v}, type: {t}}}")
    lines.append("edge:")
    for (u,v) in edges:
        lines.append(f"{u}->{v}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="单行变量数据流（每个 N.txt -> 输出到 out_dir/N.txt）")
    ap.add_argument("--in_dir", required=True, type=Path, help="输入目录，包含若干 序号.txt（每文件仅一行代码）")
    ap.add_argument("--out_dir", required=True, type=Path, help="输出目录（将生成同名 序号.txt）")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 仅处理形如 N.txt 的文件
    files = []
    for p in args.in_dir.glob("*.txt"):
        stem = p.stem
        if stem.isdigit():
            files.append(p)
    files.sort(key=lambda p: int(p.stem))

    for p in files:
        code = p.read_text(encoding="utf-8", errors="ignore").strip()
        nodes, edges = analyze_line(code)
        outp = args.out_dir / p.name
        write_one(outp, nodes, edges)

    print(f"已处理 {len(files)} 个文件，输出至：{args.out_dir}")

if __name__ == "__main__":
    main()
