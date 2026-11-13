# Neuro for Ghidra

`NeuroDecompileAll.py` sends every function of the program open in Ghidra to the local Neuro model and
brings back readable C. It works for x86-64 and ARM programs (anything Ghidra decompiles).

## Install

1. Copy `NeuroDecompileAll.py` into a Ghidra script folder (for example `~/ghidra_scripts`), or add this
   folder in Ghidra: **Window → Script Manager → Manage Script Directories**.
2. Start the model on the PC (`scripts\serve-neuro-model.ps1`, or any llama-server on port 8080).
   On another computer, change `ENDPOINT` at the top of the script to that PC's address.

## Run

Open a program, let Ghidra's auto-analysis finish, then **Tools → Neuro → Decompile whole program**
(or run it from the Script Manager). It asks:

- a folder for the results,
- whether to do only functions Ghidra knows by address (`FUN_00401a2c`) or every function,
- whether to rename those functions to the names the model chooses.

Results: one `.c` file per function plus `all.c`, in `<program>_neuro/`. Cancel stops cleanly; running
it again skips functions already saved. Functions longer than about 9,000 characters of pseudo-C are
skipped (the model was trained on functions up to ~3,000 tokens).

The model can be wrong: treat its names and code as a strong first draft, and check what matters
(recompiling a function and comparing its behaviour is the best test).
