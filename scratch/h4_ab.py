"""A/B một thay đổi NLP trên 3 phép đo: học train -> validation, giấu tên gọi, giấu câu gấp/dễ vỡ.
    python scratch/h4_ab.py FUZZY_COMMON None 20 10 50"""
import sys, subprocess, os
name, vals = sys.argv[1], sys.argv[2:]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
for v in vals:
    mod, var = name.split(".") if "." in name else ("nlp", name)
    code = f"import {mod}; {mod}.{var} = {v}\n"
    print(f"== {name} = {v}", flush=True)
    for script, extra in (("src/eval_nlp2.py", ["0", "--quick"]), ("scratch/h1_alias_holdout.py", []),
                          ("scratch/h1_alias_holdout.py", ["--phrases"])):
        src = open(script, encoding="utf-8").read()
        prog = "import sys; sys.path.insert(0, 'src'); sys.argv = [%r] + %r\n" % (script, extra) + code + src
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, env=env, encoding="utf-8").stdout
        lines = [l for l in out.splitlines() if "ĐIỂM" in l or "GIẤU" in l]
        print("   ", script.split("/")[-1], " ".join(extra), "->", lines[-1].strip() if lines else out[-300:], flush=True)
