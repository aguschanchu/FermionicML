#!/usr/bin/env python3
"""Deposit manifest tooling — freeze order Sec.4.6 rules applied at Sec.3 time.

DECLARED FORMAT (deposit-manifest-v2), the format of record for release_bundle_v2:
- A MANIFEST.sha256 contains ONLY entry lines: ^[0-9a-f]{64}(two spaces)PATH$
  PATH is POSIX-relative to the manifest's own directory, no leading './',
  no '..' segment, no absolute path, no duplicate logical path. NO comment
  lines, NO blank lines. (sha256sum -c compatible.)
- Modes are recorded per path in MODES.tsv (PATH<TAB>octal-mode), one row per
  file AND per directory of the subtree; MODES.tsv itself is listed in its
  subtree MANIFEST. The second-host re-hash compares modes, not only digests.
- Subtree scheme: every top-level subtree of a deposit root carries its own
  MANIFEST.sha256 covering every file below it (except the manifest itself).
  The root MANIFEST.sha256 lists every root-level file plus EVERY subtree
  MANIFEST.sha256 (the chain closes through the subtree manifests).
- Closure is asserted in BOTH directions (listed->present and
  present->listed), per subtree and at the root.

Commands:
  build   <root>            build subtree manifests + MODES.tsv + root manifest
  verify  <root>            strict parse + both-direction closure + digest + mode check
  parse   <manifest>        strict-parse a single manifest (refuses non-file lines)
"""
import hashlib, os, re, stat, sys

ENTRY = re.compile(r"^([0-9a-f]{64})  (.+)$")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def parse_strict(mpath):
    entries = {}
    problems = []
    with open(mpath, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            m = ENTRY.match(line)
            if not m:
                problems.append((i, "NON-FILE-LINE", line[:120]))
                continue
            d, p = m.group(1), m.group(2)
            if p.startswith("/") or p.startswith("./") or ".." in p.split("/"):
                problems.append((i, "BAD-PATH", p))
                continue
            if p in entries:
                problems.append((i, "DUPLICATE-PATH", p))
                continue
            entries[p] = d
    return entries, problems

def walk_files(root, skip=()):
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort(); fn.sort()
        for f in fn:
            full = os.path.join(dp, f)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in skip:
                continue
            out.append(rel)
    return out

def build_subtree(sub):
    files = [f for f in walk_files(sub, skip=("MANIFEST.sha256",))]
    # MODES.tsv first (it is listed in the manifest, so write before hashing)
    modes_path = os.path.join(sub, "MODES.tsv")
    rows = []
    for dp, dn, fn in os.walk(sub):
        for name in sorted(dn) + sorted(fn):
            full = os.path.join(dp, name)
            rel = os.path.relpath(full, sub).replace(os.sep, "/")
            if rel == "MODES.tsv" or rel == "MANIFEST.sha256":
                continue
            rows.append((rel, oct(stat.S_IMODE(os.lstat(full).st_mode))))
    with open(modes_path, "w") as fh:
        for rel, m in sorted(rows):
            fh.write(f"{rel}\t{m}\n")
    files = [f for f in walk_files(sub, skip=("MANIFEST.sha256",))]
    with open(os.path.join(sub, "MANIFEST.sha256"), "w") as fh:
        for rel in sorted(files):
            fh.write(f"{sha256(os.path.join(sub, rel))}  {rel}\n")
    return len(files)

def build(root):
    subs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    total = 0
    for d in subs:
        n = build_subtree(os.path.join(root, d))
        print(f"subtree {d}: {n} files manifested")
        total += n
    # root manifest: root-level files + each subtree MANIFEST.sha256
    root_files = sorted(f for f in os.listdir(root)
                        if os.path.isfile(os.path.join(root, f))
                        and f != "MANIFEST.sha256")
    with open(os.path.join(root, "MANIFEST.sha256"), "w") as fh:
        for f in root_files:
            fh.write(f"{sha256(os.path.join(root, f))}  {f}\n")
        for d in subs:
            rel = f"{d}/MANIFEST.sha256"
            fh.write(f"{sha256(os.path.join(root, rel))}  {rel}\n")
    print(f"root manifest: {len(root_files)} root files + {len(subs)} subtree manifests; {total} subtree files total")
    return 0

def verify(root):
    fail = []
    # root manifest
    rman = os.path.join(root, "MANIFEST.sha256")
    rent, rprob = parse_strict(rman)
    for i, k, t in rprob:
        fail.append(f"root manifest line {i}: {k}: {t}")
    subs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    # direction listed->present + digest, at root
    for p, d in rent.items():
        full = os.path.join(root, p)
        if not os.path.isfile(full):
            fail.append(f"root: listed-not-present: {p}")
        elif sha256(full) != d:
            fail.append(f"root: digest mismatch: {p}")
    # direction present->listed at root level (root files + subtree manifests)
    expect = set(f for f in os.listdir(root)
                 if os.path.isfile(os.path.join(root, f)) and f != "MANIFEST.sha256")
    expect |= {f"{d}/MANIFEST.sha256" for d in subs}
    for p in sorted(expect - set(rent)):
        fail.append(f"root: present-not-listed: {p}")
    # per subtree
    for d in subs:
        sub = os.path.join(root, d)
        sman = os.path.join(sub, "MANIFEST.sha256")
        if not os.path.isfile(sman):
            fail.append(f"{d}: missing subtree MANIFEST.sha256")
            continue
        ent, prob = parse_strict(sman)
        for i, k, t in prob:
            fail.append(f"{d} manifest line {i}: {k}: {t}")
        for p, dg in ent.items():
            full = os.path.join(sub, p)
            if not os.path.isfile(full):
                fail.append(f"{d}: listed-not-present: {p}")
            elif sha256(full) != dg:
                fail.append(f"{d}: digest mismatch: {p}")
        present = set(walk_files(sub, skip=("MANIFEST.sha256",)))
        for p in sorted(present - set(ent)):
            fail.append(f"{d}: present-not-listed: {p}")
        # modes
        mpath = os.path.join(sub, "MODES.tsv")
        if not os.path.isfile(mpath):
            fail.append(f"{d}: missing MODES.tsv")
        else:
            for ln, line in enumerate(open(mpath), 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    rel, mode = line.split("\t")
                except ValueError:
                    fail.append(f"{d}/MODES.tsv:{ln}: bad row")
                    continue
                full = os.path.join(sub, rel)
                if not os.path.exists(full):
                    fail.append(f"{d}/MODES.tsv:{ln}: path gone: {rel}")
                elif oct(stat.S_IMODE(os.lstat(full).st_mode)) != mode:
                    fail.append(f"{d}: MODE drift: {rel} "
                                f"{oct(stat.S_IMODE(os.lstat(full).st_mode))} != {mode}")
    # cross-boundary link scan
    for dp, dn, fn in os.walk(root):
        for name in dn + fn:
            full = os.path.join(dp, name)
            if os.path.islink(full):
                fail.append(f"SYMLINK in deposit: {full}")
            elif os.path.isfile(full) and os.lstat(full).st_nlink > 1:
                fail.append(f"HARDLINK (nlink>1) in deposit: {full}")
    for f in fail[:200]:
        print("FAIL:", f)
    print(f"verify: {'PASS' if not fail else f'{len(fail)} failures'}")
    return 0 if not fail else 1

if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "build":
        sys.exit(build(arg))
    elif cmd == "verify":
        sys.exit(verify(arg))
    elif cmd == "parse":
        ent, prob = parse_strict(arg)
        print(f"{len(ent)} entries, {len(prob)} refused lines")
        for i, k, t in prob[:40]:
            print(f"  line {i}: {k}: {t}")
        sys.exit(0 if not prob else 1)
