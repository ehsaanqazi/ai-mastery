This quiz covers the whole Python Programming module, from variables to async. Answer every question to see your score. Your best result is saved, and you can retry as often as you like. Aim for at least 15 out of 20 before moving on to Developer Tools.

```quiz
[
  {
    "q": "What does `print(type(7 / 2))` display?",
    "options": ["`<class 'int'>`", "`<class 'float'>`", "`<class 'str'>`", "`<class 'decimal'>`"],
    "answer": 1,
    "why": "The `/` operator always performs true division and returns a float."
  },
  {
    "q": "What is `23 // 5` and `23 % 5`?",
    "options": ["`4` and `3`", "`4.6` and `3`", "`5` and `2`", "`3` and `4`"],
    "answer": 0,
    "why": "23 divided by 5 is 4 with a remainder of 3."
  },
  {
    "q": "Which value is falsy?",
    "options": ["`\"0\"`", "`[0]`", "`0.0`", "`\" \"`"],
    "answer": 2,
    "why": "Zero of any numeric type is falsy. A non-empty string or list is truthy whatever it contains."
  },
  {
    "q": "How many times does `print` run in `for i in range(1, 10, 3): print(i)`?",
    "options": ["2", "3", "4", "9"],
    "answer": 1,
    "why": "The values are 1, 4 and 7. The next would be 10, which is excluded."
  },
  {
    "q": "What does `[1, 2, 3, 4, 5][-2:]` evaluate to?",
    "options": ["`[4, 5]`", "`[1, 2, 3]`", "`[5, 4]`", "`[3, 4, 5]`"],
    "answer": 0,
    "why": "A negative start counts from the end, so the slice takes the last two items."
  },
  {
    "q": "After `a = [1, 2]`, `b = a.copy()`, `b.append(3)`, what is `a`?",
    "options": ["`[1, 2, 3]`", "`[1, 2]`", "`[3]`", "`None`"],
    "answer": 1,
    "why": "`copy()` creates a separate list, so changes to `b` do not affect `a`."
  },
  {
    "q": "Which collection would you use to map each word to its count?",
    "options": ["List", "Tuple", "Set", "Dictionary"],
    "answer": 3,
    "why": "A dictionary stores key-value pairs and looks values up by key."
  },
  {
    "q": "What does this function return when called as `f(4)`?  `def f(n): print(n * 2)`",
    "options": ["`8`", "`None`", "`\"8\"`", "It raises an error"],
    "answer": 1,
    "why": "It prints 8 but has no `return` statement, so the returned value is `None`."
  },
  {
    "q": "What is `sorted([\"pear\", \"fig\", \"banana\"], key=len)`?",
    "options": [
      "`['banana', 'fig', 'pear']`",
      "`['fig', 'pear', 'banana']`",
      "`['pear', 'fig', 'banana']`",
      "`[3, 4, 6]`"
    ],
    "answer": 1,
    "why": "The `key` function is applied to each item for comparison, so the words are ordered by length."
  },
  {
    "q": "What does `\" Hello World \".strip().lower().split()` produce?",
    "options": ["`['hello', 'world']`", "`'hello world'`", "`['Hello', 'World']`", "`['', 'hello', 'world', '']`"],
    "answer": 0,
    "why": "The whitespace is stripped, the text lowercased, then split on whitespace into a list."
  },
  {
    "q": "What is `[n for n in range(6) if n % 2 == 0]`?",
    "options": ["`[1, 3, 5]`", "`[0, 2, 4]`", "`[0, 2, 4, 6]`", "`[2, 4]`"],
    "answer": 1,
    "why": "`range(6)` gives 0 to 5, and the filter keeps the even values."
  },
  {
    "q": "Which mode opens a file for adding text to the end without erasing it?",
    "options": ["`\"r\"`", "`\"w\"`", "`\"a\"`", "`\"x\"`"],
    "answer": 2,
    "why": "Append mode writes after the existing content. Write mode truncates the file first."
  },
  {
    "q": "Which exception does `{\"a\": 1}[\"b\"]` raise?",
    "options": ["`IndexError`", "`KeyError`", "`ValueError`", "`AttributeError`"],
    "answer": 1,
    "why": "Looking up a missing dictionary key with square brackets raises `KeyError`."
  },
  {
    "q": "In `class Dog(Animal):`, what is `Animal`?",
    "options": ["An instance of Dog", "The parent class that Dog inherits from", "A method of Dog", "A module"],
    "answer": 1,
    "why": "The class in parentheses is the superclass. `Dog` inherits its attributes and methods."
  },
  {
    "q": "Which method controls what `print(obj)` displays for your own class?",
    "options": ["`__init__`", "`__repr__`", "`__len__`", "`__print__`"],
    "answer": 1,
    "why": "`print` uses `__str__` if defined and otherwise falls back to `__repr__`."
  },
  {
    "q": "What is the purpose of a virtual environment?",
    "options": [
      "To run Python in the browser",
      "To isolate a project's packages from other projects",
      "To make code run on a GPU",
      "To encrypt source code"
    ],
    "answer": 1,
    "why": "Each environment has its own installed packages, preventing version conflicts between projects."
  },
  {
    "q": "What does the return annotation in `def find(id: int) -> str | None:` mean?",
    "options": [
      "The function always returns a string",
      "The function returns a string or `None`",
      "The function returns a list of strings",
      "Python will raise an error if `None` is returned"
    ],
    "answer": 1,
    "why": "The `|` creates a union type. Callers should handle the `None` case."
  },
  {
    "q": "What does `yield` do inside a function?",
    "options": [
      "Ends the function permanently, like `return`",
      "Produces a value and pauses the function so it can resume later",
      "Raises an exception",
      "Imports a module lazily"
    ],
    "answer": 1,
    "why": "`yield` turns the function into a generator, which hands back values one at a time and keeps its state between them."
  },
  {
    "q": "Four coroutines each wait 3 seconds on a network call. How long does `await asyncio.gather(...)` over all four take?",
    "options": ["About 3 seconds", "About 6 seconds", "About 12 seconds", "It depends on CPU cores"],
    "answer": 0,
    "why": "The waits overlap, so the total is roughly the longest single wait."
  },
  {
    "q": "Which call turns the text of an API response into Python dictionaries and lists?",
    "options": ["`json.dumps(text)`", "`json.loads(text)`", "`csv.reader(text)`", "`str(text)`"],
    "answer": 1,
    "why": "`loads` parses a JSON string into Python objects. `dumps` does the reverse."
  }
]
```
