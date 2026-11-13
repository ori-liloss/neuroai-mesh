# Send every function of the open program to the local Neuro model and bring back readable C.
# @category Neuro
# @menupath Tools.Neuro.Decompile whole program
#
# For each function: Ghidra's own decompiler output (pseudo-C) goes to the model, which answers with
# readable C, as it was trained to (Ghidra/IDA pseudo-C -> source). The script then
#   - saves every answer to <program>_neuro/<address>_<name>.c plus all functions in all.c,
#   - optionally renames functions Ghidra only knows by address (FUN_00401a2c) to the name the
#     model chose in its answer, and puts a short comment on them,
#   - stops cleanly when you press Cancel; running it again skips functions already saved.
# The model answers on an OpenAI-style endpoint (llama-server: http://127.0.0.1:8080). Works with
# Ghidra's Jython (Python 2.7) and with PyGhidra (Python 3). Use it on programs you may analyze.
import json
import os
import re
import time

try:  # Python 3 (PyGhidra)
    from urllib.request import Request, urlopen
except ImportError:  # Jython 2.7
    from urllib2 import Request, urlopen

from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.symbol import SourceType

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
MAX_PSEUDO_CHARS = 9000      # the model was trained on functions up to ~3,000 tokens
TIMEOUT_S = 300
SYSTEM = "You are Neuro, an expert in reverse engineering, C and C++."
DEFAULT_NAME = re.compile(r"^(FUN|thunk_FUN|sub|LAB)_[0-9a-fA-F]+$")


def ask_model(prompt):
    body = json.dumps({
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.2, "max_tokens": 3072, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
    reply = json.loads(urlopen(request, timeout=TIMEOUT_S).read().decode("utf-8"))
    return reply["choices"][0]["message"]["content"]


def code_of(answer):
    match = re.search(r"```(?:c|cpp|c\+\+)?\s*\n(.*?)```", answer, re.S)
    return (match.group(1) if match else answer).strip()


def name_in(code):
    """The function name the model gave in its definition: `type name(args) {`."""
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", code):
        name = match.group(1)
        if name not in ("if", "for", "while", "switch", "return", "sizeof"):
            return name
    return None


def safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)[:80]


def main():
    program = currentProgram
    arch = program.getLanguage().getProcessor().toString()
    out_dir = askDirectory("Folder for the recovered C files", "Choose").getAbsolutePath()
    folder = os.path.join(out_dir, safe(program.getName()) + "_neuro")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    only_unnamed = askYesNo("Neuro", "Only functions Ghidra knows by address (FUN_...)?\n\nNo = every function.")
    rename = askYesNo("Neuro", "Rename those functions to the names the model chooses?")
    decompiler = DecompInterface()
    decompiler.openProgram(program)
    functions = [f for f in program.getFunctionManager().getFunctions(True)
                 if not f.isExternal() and not f.isThunk()]
    if only_unnamed:
        functions = [f for f in functions if DEFAULT_NAME.match(f.getName())]
    monitor.initialize(len(functions))
    done = skipped = failed = 0
    started = time.time()
    combined = open(os.path.join(folder, "all.c"), "a")
    try:
        for index, function in enumerate(functions):
            monitor.checkCancelled()
            monitor.setProgress(index)
            address = function.getEntryPoint().toString()
            old_name = function.getName()
            monitor.setMessage("Neuro: %s (%d of %d)" % (old_name, index + 1, len(functions)))
            target = os.path.join(folder, "%s_%s.c" % (safe(address), safe(old_name)))
            if os.path.exists(target):
                skipped += 1
                continue
            result = decompiler.decompileFunction(function, 60, monitor)
            if not result or not result.decompileCompleted():
                failed += 1
                continue
            pseudo = result.getDecompiledFunction().getC()
            if len(pseudo) > MAX_PSEUDO_CHARS:
                skipped += 1
                continue
            prompt = ("This is Ghidra pseudo-C of the function `%s` from a %s binary. Rewrite it as the original, "
                      "readable C source: real types and names, natural control flow, no decompiler casts or "
                      "temporaries. Keep the behaviour exact.\n\n```c\n%s\n```" % (old_name, arch, pseudo.strip()))
            try:
                code = code_of(ask_model(prompt))
            except Exception as error:  # the model may be busy or down: note it and go on
                println("Neuro: %s failed: %s" % (old_name, error))
                failed += 1
                continue
            new_name = name_in(code)
            header = "/* %s at %s, recovered by Neuro from Ghidra's output */\n" % (old_name, address)
            handle = open(target, "w")
            handle.write(header + code + "\n")
            handle.close()
            combined.write(header + code + "\n\n")
            combined.flush()
            if rename and new_name and DEFAULT_NAME.match(old_name) and new_name != old_name:
                try:
                    function.setName(new_name, SourceType.USER_DEFINED)
                    function.setComment("Named by Neuro (was %s). Recovered C: %s" % (old_name, os.path.basename(target)))
                except Exception as error:  # a name already in use, or an invalid one
                    println("Neuro: could not rename %s to %s: %s" % (old_name, new_name, error))
            done += 1
    finally:
        combined.close()
        decompiler.dispose()
    minutes = (time.time() - started) / 60.0
    println("Neuro: %d functions recovered, %d skipped (already saved or too long), %d failed, in %.1f min. Files: %s"
            % (done, skipped, failed, minutes, folder))


main()
