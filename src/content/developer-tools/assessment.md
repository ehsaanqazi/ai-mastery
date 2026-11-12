This quiz covers the whole Developer Tools module: Linux, the terminal, Git, VS Code, Jupyter, debugging, environments and development practices. Answer every question to see your score. Your best result is saved. Aim for at least 12 out of 15 before moving on to Phase 2.

```quiz
[
  {
    "q": "In Linux, what does `~` refer to?",
    "options": ["The root directory", "The current directory", "Your home directory", "The previous directory"],
    "answer": 2,
    "why": "`~` expands to the current user's home directory, such as `/home/amina`."
  },
  {
    "q": "What does `chmod 600 ~/.ssh/id_ed25519` achieve?",
    "options": [
      "Makes the key executable by everyone",
      "Lets only the owner read and write the key",
      "Deletes the key",
      "Shares the key with the group"
    ],
    "answer": 1,
    "why": "6 is read plus write for the owner; the two zeros remove all access for group and others."
  },
  {
    "q": "Which command counts the lines in a file containing the word `ERROR`?",
    "options": ["`grep ERROR app.log | wc -l`", "`cat ERROR app.log`", "`wc ERROR app.log`", "`tail -f ERROR`"],
    "answer": 0,
    "why": "`grep` keeps the matching lines and the pipe passes them to `wc -l`, which counts them."
  },
  {
    "q": "What does `python train.py > out.log 2>&1` do?",
    "options": [
      "Runs the script twice",
      "Sends both normal output and errors to `out.log`",
      "Sends only errors to `out.log`",
      "Appends to `out.log`"
    ],
    "answer": 1,
    "why": "`>` redirects standard output, and `2>&1` sends standard error to the same place."
  },
  {
    "q": "What is the correct order of the everyday Git workflow?",
    "options": [
      "commit, add, edit",
      "edit, add, commit",
      "add, edit, push",
      "push, commit, add"
    ],
    "answer": 1,
    "why": "You change files, stage the changes you want with `git add`, then record them with `git commit`."
  },
  {
    "q": "What do the markers `<<<<<<<`, `=======` and `>>>>>>>` in a file indicate?",
    "options": ["A syntax error", "A merge conflict", "A commented-out block", "A Git hook"],
    "answer": 1,
    "why": "Git inserts them when two branches changed the same lines and it cannot choose automatically."
  },
  {
    "q": "Which of these belongs in `.gitignore`?",
    "options": ["`README.md`", "`pyproject.toml`", "`.env`", "`src/`"],
    "answer": 2,
    "why": "`.env` holds secrets and must never be committed. The others are part of the project."
  },
  {
    "q": "In VS Code, imports show as unresolved even though the packages are installed. What should you check first?",
    "options": [
      "The selected Python interpreter",
      "The colour theme",
      "The Git branch",
      "The font size"
    ],
    "answer": 0,
    "why": "The editor analyses code with the selected interpreter, which must be the project's environment."
  },
  {
    "q": "Why can a Jupyter notebook work for you but fail when a colleague runs it from the top?",
    "options": [
      "Notebooks only run on one machine",
      "Your kernel kept variables from cells that were later deleted or edited",
      "Markdown cells are executed differently",
      "Jupyter randomises execution order"
    ],
    "answer": 1,
    "why": "The kernel's memory can hold state that the saved notebook no longer creates. Restart and run all to check."
  },
  {
    "q": "What does `%timeit` do in a notebook?",
    "options": [
      "Sets a time limit on a cell",
      "Times a statement accurately over many runs",
      "Prints the current time",
      "Schedules a cell to run later"
    ],
    "answer": 1,
    "why": "`%timeit` repeats the code many times and reports the mean and spread."
  },
  {
    "q": "A function returns a wrong number but raises no error. Which tool helps most?",
    "options": [
      "A debugger, stepping through and watching variables",
      "Reinstalling Python",
      "Adding `try`/`except`",
      "Deleting the virtual environment"
    ],
    "answer": 0,
    "why": "Stepping through shows exactly where a value first diverges from what you expected."
  },
  {
    "q": "What does calling `breakpoint()` in your code do?",
    "options": [
      "Ends the program",
      "Pauses execution and opens the debugger at that line",
      "Prints a stack trace and continues",
      "Raises an exception"
    ],
    "answer": 1,
    "why": "It drops you into `pdb`, where you can inspect variables and step forward."
  },
  {
    "q": "Which file pins every package to an exact version in a uv project?",
    "options": ["`pyproject.toml`", "`.python-version`", "`uv.lock`", "`.venv/pyvenv.cfg`"],
    "answer": 2,
    "why": "The lock file records exact versions of all direct and transitive dependencies."
  },
  {
    "q": "Where should an API key used by your application be stored?",
    "options": [
      "As a constant at the top of `config.py`",
      "In an environment variable, kept out of Git",
      "In a notebook cell",
      "In the README"
    ],
    "answer": 1,
    "why": "Secrets in source files or notebooks end up in version control and get leaked."
  },
  {
    "q": "What is a regression test?",
    "options": [
      "A test for linear regression models",
      "A test that reproduces a fixed bug so it cannot return unnoticed",
      "A test that runs slowly",
      "A test that is skipped in production"
    ],
    "answer": 1,
    "why": "Writing a failing test for a bug before fixing it guards against that bug coming back."
  }
]
```
