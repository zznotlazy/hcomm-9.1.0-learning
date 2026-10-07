"""Build static, source-linked learning evidence. Does not execute HCCL.

Run from any directory. Only writes generated documentation under this folder.
The lexer preserves offsets while masking comments/strings; this is a source
indexer, not a C++ compiler or a proof of branch reachability.
"""
from pathlib import Path
import os
import hashlib
import json
import re
import subprocess
from functools import lru_cache

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
BASE = ROOT / 'src/legacy/ascend910'
EVIDENCE = OUT / 'evidence'
FILES = set()
ISSUES = []
UNRESOLVED = []


def sorted_paths(paths):
    # Explicit lexical ordering keeps generated output equal across host OSes.
    return sorted(paths, key=lambda p: p.relative_to(ROOT).as_posix())


def read(path):
    FILES.add(path)
    return path.read_text(encoding='utf-8-sig')


def link(path, line=1, label=None):
    return f'[{label or path.name}](repo-source:{path.relative_to(ROOT).as_posix()}#L{line})'


def mask_cpp(text):
    # Keep newline and byte-character offsets, ignore braces in comments/literals.
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), text)


@lru_cache(maxsize=None)
def functions(path):
    text = read(path)
    masked = mask_cpp(text)
    pattern = re.compile(
        r'^[ \t]*(?:[\w:<>,*&~]+[ \t]+)+(\w+(?:::[~\w]+)?)\s*'
        r'\([^;{}]*\)\s*(?:const\s*)?\{', re.M)
    result = []
    for m in pattern.finditer(masked):
        start = m.start()
        b = masked.find('{', m.start(), m.end())
        depth, end = 1, b + 1
        while depth and end < len(masked):
            depth += (masked[end] == '{') - (masked[end] == '}')
            end += 1
        if depth:
            raise ValueError(f'Unbalanced function {path}:{m[1]}')
        result.append((m[1], text[:start].count('\n') + 1, text[start:end].strip()))
    return result


def excerpts(path, chosen):
    parts = []
    for name, line, body in chosen:
        parts.append(f'### {name}\n\n{link(path, line)}\n\n```cpp\n{body}\n```\n')
    return '\n'.join(parts)


def git(*args):
    p = subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def resolve_anchors():
    for path in sorted_paths(OUT.glob('*.md')):
        text = path.read_text(encoding='utf-8')
        def replace(m):
            base = BASE if m[1] == 'src' else ROOT
            source = base / m[2]
            content = read(source)
            position = content.find(m[3])
            if position < 0:
                raise ValueError(f'Missing anchor in {source}: {m[3]}')
            return '](repo-source:' + source.relative_to(ROOT).as_posix() + '#L' + str(content[:position].count('\n') + 1) + ')'
        updated = re.sub(r'\]\(@(src|repo):([^|\n]+)\|([^\n]+?)\)', replace, text)
        if updated != text:
            path.write_text(updated, encoding='utf-8', newline='\n')


EVIDENCE.mkdir(exist_ok=True)
resolve_anchors()

all_cc = sorted_paths((BASE / 'algorithm').rglob('*.cc'))
registrations, templates = {}, {}
for path in all_cc:
    text = read(path)
    for m in re.finditer(r'REGISTER_EXEC\(\s*"([^"]+)"\s*,\s*\w+\s*,\s*(\w+)\s*\)', text):
        name, cls = m.groups()
        entry = (path, text[:m.start()].count('\n') + 1, cls)
        if name in registrations:
            ISSUES.append(f'Duplicate executor registry {name}')
        registrations[name] = entry
    for m in re.finditer(r'REGISTER_TEMPLATE\(\s*TemplateType::(\w+)\s*,\s*(\w+)\s*\)', text):
        templates.setdefault(m[1], []).append((path, text[:m.start()].count('\n') + 1, m[2]))

classes = {}
for path in sorted_paths((BASE / 'algorithm').rglob('*.h')):
    text = read(path)
    for m in re.finditer(r'class\s+(\w+)\s*:\s*public\s+(\w+)', text):
        classes.setdefault(m[1], (m[2], path))

index = ['# 源码证据索引\n',
         '自动从当前快照生成。主文档负责解释A2可达路径；本索引保留有序条件的完整函数。'
         '执行器和模板表是源码出现清单，不自动证明全部分支在A2可达。\n',
         '## 注册命令清单\n', '| 命令 | 选择器 | 注册位置 |', '|---|---|---|']
commands = []
op_dir = BASE / 'algorithm/impl/operator'
for path in sorted_paths(op_dir.glob('*_operator.cc')):
    text = read(path)
    for m in re.finditer(r'REGISTER_OP\(\s*HcclCMDType::(\w+)\s*,\s*\w+\s*,\s*(\w+)\s*\)', text):
        commands.append(m[1])
        index.append(f'| {m[1]} | {m[2]} | {link(path, text[:m.start()].count(chr(10))+1)} |')

selections = {}
manifest = {}
common = op_dir / 'coll_alg_operator.cc'
chosen = [f for f in functions(common) if any(k in f[0] for k in
          ['SelectAlg', 'SelectAlgo', 'GetDefaultAlgo', 'CalcContext', 'SetTopoAttr',
           'SetAlgoAttr', 'SetExecutorAttr', 'SingleMesh', 'MultiMesh', 'Is2U2P',
           'CalcResRequest', 'Orchestrate', 'AHCAlg', 'IsNeedStrictMode'])
          and '91093' not in f[0]]
public_evidence = '# 公共选择函数\n\n' + excerpts(common, chosen)
for relative in ['algorithm/impl/alg_configurator.cc', 'algorithm/impl/alg_env_config.cc',
                 'algorithm/impl/coll_alg_utils.cc', 'platform/common/device_capacity.cc']:
    path = BASE / relative
    chosen = [f for f in functions(path) if not any(k in f[0] for k in ['91093', '310P', 'ARS', 'Retry'])]
    public_evidence += f'\n## {path.name}\n\n' + excerpts(path, chosen)
(EVIDENCE / 'common.md').write_text(public_evidence, encoding='utf-8', newline='\n')
index += ['\n## 公共条件原文\n', '[公共函数证据](evidence/common.md)\n']

for path in sorted_paths(op_dir.glob('*_operator.cc')):
    if path.name == 'coll_alg_operator.cc':
        continue
    stem = path.stem.replace('_operator', '')
    chosen = [f for f in functions(path)
              if not any(k in f[0] for k in ['91093', '310P', '910A', 'Mix'])
              and any(k in f[0] for k in ['Select', 'Condition', 'GetCountType', 'CopyMode'])]
    # Includes main entry/device dispatcher; out-of-scope branches remain visible there.
    if not chosen:
        chosen = functions(path)
    target = EVIDENCE / f'{stem}.md'
    evidence_text = (f'# {stem} 选择证据\n\n'
                     '按函数和源码顺序阅读；主入口中保留其他芯片的分派，A2仅跟随910B路径。\n\n'
                     + excerpts(path, chosen))
    names = sorted(set(re.findall(r'algName\s*=\s*"([^"]+)"', '\n'.join(f[2] for f in chosen))))
    # ternary assignment alternatives are also candidates.
    for f in chosen:
        for line in f[2].splitlines():
            if 'algName =' in line:
                names += re.findall(r'"([^"]+)"', line)
    names = sorted(set(names))
    selections[stem] = names
    index += [f'\n## {stem}\n', f'[完整选择函数](evidence/{stem}.md)\n',
              '| 候选名称 | 注册 / 类 | 基类链 | 内部证据 |', '|---|---|---|---|']
    for name in names:
        if name not in registrations:
            UNRESOLVED.append(f'{stem}: {name}')
            if name != 'BroadCast4pRingExecutor':
                ISSUES.append(f'Unresolved candidate {stem}: {name}')
            index.append(f'| {name} | 未解析 | — | — |')
            continue
        file, line, cls = registrations[name]
        chain = []
        cur = cls
        source_files = [file]
        seen = set()
        while cur in classes and cur not in seen:
            seen.add(cur)
            parent, header = classes[cur]
            chain.append(parent)
            implementation = header.with_suffix('.cc')
            if implementation.exists() and implementation not in source_files:
                source_files.append(implementation)
            cur = parent
        executor_evidence = EVIDENCE / f'executor_{name}.md'
        executor_text = f'# {name}\n\n注册：{link(file,line)}；类 `{cls}`。\n\n'
        executor_text += '共享代码可能含其他芯片分支。模板清单为出现位置；按完整函数中的设备/层次/子域条件确认实际执行。\n\n'
        refs = {}
        for sf in source_files:
            fn = functions(sf)
            relevant = [f for f in fn if any(k in f[2] for k in
                        ['TemplateType::', 'GetAlgTemplate', 'Aiv', 'AIV', 'algType_', 'RunLoop'])
                        or any(k in f[0] for k in ['KernelRun', 'Orchestrate', 'Select', 'LoopMaxCount'])]
            shared_name = sf.relative_to(BASE).as_posix().replace('/', '__').replace('.cc', '.md')
            shared_dir = EVIDENCE / 'sources'
            shared_dir.mkdir(exist_ok=True)
            shared_path = shared_dir / shared_name
            shared_path.write_text(f'# {sf.name}\n\n共享执行器函数；按设备条件确认A2分支。\n\n' + excerpts(sf, relevant), encoding='utf-8', newline='\n')
            executor_text += f'## {sf.name}\n\n[完整函数与内部条件](sources/{shared_name})\n\n'
            text = read(sf)
            for m in re.finditer(r'TemplateType::(TEMPLATE_\w+)', text):
                refs.setdefault(m[1], []).append((sf, text[:m.start()].count('\n') + 1))
        executor_text += '## 模板注册与出现位置\n\n| 模板 | 首个出现 | 实现注册 |\n|---|---|---|\n'
        for tpl, occurrences in sorted(refs.items()):
            matches = templates.get(tpl, [])
            register_links = ', '.join(link(p, n, c) for p, n, c in matches) or '未解析注册（可能非传统注册模板）'
            executor_text += f'| {tpl} | {link(*occurrences[0])} | {register_links} |\n'
        executor_evidence.write_text(executor_text, encoding='utf-8', newline='\n')
        index.append(f'| {name} | {link(file,line,cls)} | {" → ".join(chain) or "—"} | [函数与模板](evidence/executor_{name}.md) |')
        manifest[name] = {'class': cls, 'registry': file.relative_to(ROOT).as_posix(),
                          'line': line, 'parents': chain, 'templates': sorted(refs)}
    target.write_text(evidence_text, encoding='utf-8', newline='\n')

index += ['\n## AIV 内核入口\n',
          'AIV不一定使用传统模板注册；按设备宏和已选择执行器进入以下分派入口。\n']
for path in sorted_paths((BASE / 'algorithm/base/alg_aiv_template').rglob('aiv_*_op.h')):
    index.append(f'- {link(path)}')

index += ['\n## 阈值定义\n', '| 名称 | 定义 | 位置 |', '|---|---|---|']
constant_paths = [BASE/'algorithm/pub_inc/hccl_aiv.h', BASE/'algorithm/pub_inc/common.h',
                  BASE/'algorithm/pub_inc/op_context.h',
                  BASE/'pub_inc/hccl_common.h', BASE/'algorithm/impl/coll_alg_utils.h',
                  BASE/'algorithm/base/alg_template/alg_template_base_pub.h',
                  common, op_dir/'alltoall_operator.cc', op_dir/'broadcast_operator.cc']
for path in constant_paths:
    for i, line in enumerate(read(path).splitlines(), 1):
        if 'constexpr' in line and any(k in line for k in ['AIV_', 'COUNT_', 'FFTS', 'PIPELINE', 'RANK_SIZE', 'SLICE_ALIGN', 'ALLTOALLV_', 'SERVER_NUM', 'NHR_']):
            parts = line.split('=',1)
            if len(parts) == 2:
                index.append(f'| `{parts[0].split()[-1]}` | `{parts[1].split(";")[0].strip()}` | {link(path,i)} |')

index += ['\n## 已有测试入口（未执行）\n', '| 测试名 | 源码 |', '|---|---|']
for relative in ['test/ut/inter/comm/ut_coll_alg_operator.cc', 'test/ut/inter/comm/ut_allgather_common.cc',
                 'test/ut/impl/ut_hccl_impl_new.cc']:
    path = ROOT / relative
    text = read(path)
    for m in re.finditer(r'TEST_F\(\s*(\w+)\s*,\s*(\w+)\s*\)',text):
        if relative.endswith('ut_hccl_impl_new.cc') and 'SelectAlg' not in m[2]:
            continue
        index.append(f'| {m[1]}.{m[2]} | {link(path,text[:m.start()].count(chr(10))+1)} |')

index += ['\n## 索引限制\n',
          '候选来自人工限定的选择函数；主入口保留非A2分派，Mix分支另见主文档。继承链索引用于跟踪公共模板，'
          '模板出现表含共享函数内其他芯片分支，不能代替可达性判断。'
          '没有在这里运行C++预处理、生产选择器、CANN测试或性能测量。\n',
          '\n## 未解析注册项\n',
          'BroadCast4pRingExecutor：源码选择器存在这个字符串，但当前传统注册表未找到相同名称。'
          '不能宣称可成功执行；主文档记录正常A2拓扑来源与此异常候选。\n']
(OUT/'source_index.md').write_text('\n'.join(index)+'\n',encoding='utf-8', newline='\n')
(OUT/'selection_manifest.json').write_text(json.dumps({'commands': commands, 'candidates': selections,
                                                      'executors': manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')

# Resolve repository markers relative to each document, for GitHub and Linux.
for doc in sorted_paths(OUT.rglob('*.md')):
    content = doc.read_text(encoding='utf-8')
    def portable(m):
        source = ROOT / m[1]
        return '](' + os.path.relpath(source, doc.parent).replace('\\', '/') + '#L' + m[2] + ')'
    content = re.sub(r'\]\(repo-source:([^\n)]+)#L(\d+)\)', portable, content)
    doc.write_text(content, encoding='utf-8', newline='\n')

# Create the referenced baseline before link validation; enrich it afterwards.
(OUT/'baseline.json').write_text('{}\n',encoding='utf-8', newline='\n')
link_count = 0
for doc in sorted_paths(OUT.rglob('*.md')):
    content = re.sub(r'```[^\n]*\n[\s\S]*?```', '', doc.read_text(encoding='utf-8'))
    if any(marker in content for marker in ['@src:', '@repo:', 'repo-source:']):
        ISSUES.append(f'Unresolved anchor in {doc.relative_to(ROOT)}')
    for m in re.finditer(r'\]\(([^\n)]+)\)',content):
        dest = m[1]
        if dest.startswith(('http:', 'https:', '#')):
            continue
        location, _, fragment = dest.partition('#')
        if re.match(r'^[A-Za-z]:', location) or location.startswith('/'):
            ISSUES.append(f'Non-portable link in {doc.relative_to(ROOT)}: {dest}')
            continue
        path = (doc.parent / location).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            ISSUES.append(f'Missing/escaping link {doc.name}: {dest}')
            continue
        link_count += 1
        if path.suffix in {'.cc','.h'}:
            source = read(path)
            lm = re.fullmatch(r'L(\d+)',fragment)
            if lm and not (1 <= int(lm[1]) <= len(source.splitlines())):
                ISSUES.append(f'Invalid line {dest}')

expected = {'ALLGATHER','ALLGATHER_V','ALLREDUCE','REDUCE_SCATTER','REDUCE_SCATTER_V',
            'BROADCAST','REDUCE','SCATTER','ALLTOALL','ALLTOALLV','ALLTOALLVC',
            'SEND','RECEIVE','BATCH_SEND_RECV','BATCH_WRITE'}
if set(commands) != {'HCCL_CMD_'+x for x in expected}:
    ISSUES.append('Registered command coverage changed')

provenance = json.loads((OUT/'source_provenance.json').read_text(encoding='utf-8'))
baseline = {'repository': 'https://github.com/zznotlazy/hcomm-9.1.0-learning',
            'upstream': {key: provenance[key] for key in ['upstream_url','upstream_branch','upstream_commit']},
            'analysis_date': '2026-10-07', 'scope': 'A2/DEV_TYPE_910B traditional selection path',
            'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted_paths(FILES) if p.is_relative_to(ROOT)}}
(OUT/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
result = {'registered_commands': len(commands), 'candidate_executors': len(manifest),
          'markdown_links_checked': link_count, 'source_files_hashed': len(baseline['files']),
          'issues': ISSUES, 'unresolved_registry_findings': UNRESOLVED,
          'validation': 'Static indexing and traceability only; HCCL not executed'}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8', newline='\n')
print(json.dumps(result,ensure_ascii=False))
if ISSUES:
    raise SystemExit(1)
