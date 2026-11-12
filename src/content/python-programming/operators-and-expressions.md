An **expression** is any piece of code that produces a value: `2 + 3`, `len(name)`, `score > 0.5`. **Operators** are the symbols that combine values into expressions. This lesson covers the operators you will use daily and the rules Python follows when several appear together.

## Arithmetic operators

| Operator | Meaning | Example | Result |
| --- | --- | --- | --- |
| `+` | Addition | `7 + 2` | `9` |
| `-` | Subtraction | `7 - 2` | `5` |
| `*` | Multiplication | `7 * 2` | `14` |
| `/` | True division | `7 / 2` | `3.5` |
| `//` | Floor division | `7 // 2` | `3` |
| `%` | Modulo (remainder) | `7 % 2` | `1` |
| `**` | Power | `7 ** 2` | `49` |

Two details catch people out. `/` always returns a float, even when the division is exact: `6 / 2` is `3.0`. And `//` rounds down towards negative infinity, so `-7 // 2` is `-4`, not `-3`.

Floor division and modulo are two halves of the same operation. For any integers $a$ and $b$ with $b \neq 0$:

$$
a = b \cdot (a \mathbin{//} b) + (a \bmod b)
$$

With $a = 17$ and $b = 5$: $17 = 5 \cdot 3 + 2$.

```python
total_samples = 1050
batch_size = 32

full_batches = total_samples // batch_size
leftover = total_samples % batch_size
print(full_batches, leftover)
```

```output
32 26
```

That calculation, splitting a dataset into batches, is one you will do for real when training models.

### Common uses of modulo

```python
print(10 % 2 == 0)     # is 10 even?
print(37 % 10)         # last digit
print((23 + 5) % 24)   # clock arithmetic: 5 hours after 23:00
```

```output
True
7
4
```

## Comparison operators

Comparisons produce a boolean.

| Operator | Meaning |
| --- | --- |
| `==` | Equal to |
| `!=` | Not equal to |
| `<`, `>` | Less than, greater than |
| `<=`, `>=` | Less or equal, greater or equal |

```python
accuracy = 0.92
print(accuracy > 0.9)
print(accuracy == 1.0)
print(0.5 <= accuracy < 0.95)   # comparisons can be chained
```

```output
True
False
True
```

> [!WARNING]
> `=` assigns and `==` compares. Writing `if x = 5:` is a `SyntaxError`. Mixing the two up is one of the most common beginner mistakes.

## Logical operators

`and`, `or` and `not` combine booleans.

| `a` | `b` | `a and b` | `a or b` | `not a` |
| --- | --- | --- | --- | --- |
| True | True | True | True | False |
| True | False | False | True | False |
| False | True | False | True | True |
| False | False | False | False | True |

```python
has_gpu = True
memory_gb = 8

can_train = has_gpu and memory_gb >= 16
print(can_train)
print(not can_train)
```

```output
False
True
```

### Short-circuit evaluation

Python stops evaluating as soon as the answer is known. In `a and b`, if `a` is false, `b` is never looked at. In `a or b`, if `a` is true, `b` is skipped.

```python
items = []
if items and items[0] == "start":   # items[0] would crash on an empty list
    print("found")
```

The second condition is only evaluated when the list is non-empty, so this is safe. This pattern, a guard followed by the real check, appears everywhere.

`or` also gives you a compact way to supply a default:

```python
user_input = ""
name = user_input or "Anonymous"
print(name)
```

```output
Anonymous
```

## Assignment operators

Updating a variable based on its own value is so common that it has shorthand:

```python
loss = 10.0
loss -= 2.5     # same as loss = loss - 2.5
loss *= 0.5
print(loss)

steps = 0
steps += 1
print(steps)
```

```output
3.75
1
```

Python has no `++` operator. Use `+= 1`.

## Membership and identity

`in` tests whether a value is inside a collection or string. `is` tests whether two names point to the very same object.

```python
print("py" in "python")
print(3 in [1, 2, 3])
print("x" not in "abc")

result = None
print(result is None)
```

```output
True
True
True
True
```

> [!KEY]
> Use `==` to ask "do these have the same value?" and `is` to ask "is this the same object?". In practice, reserve `is` for `None`, `True` and `False`.

## Operator precedence

When an expression has several operators, Python applies them in a fixed order, highest first:

1. Parentheses `( )`
2. Power `**`
3. Unary minus `-x`
4. `*`, `/`, `//`, `%`
5. `+`, `-`
6. Comparisons, `in`, `is`
7. `not`
8. `and`
9. `or`

```python
print(2 + 3 * 4)
print((2 + 3) * 4)
print(-2 ** 2)
print(True or False and False)
```

```output
14
20
-4
True
```

`-2 ** 2` is `-4` because power binds tighter than the minus sign. The last line is `True` because `and` runs before `or`.

> [!TIP]
> You do not need to memorise the table. When an expression is not obvious at a glance, add parentheses. They cost nothing and remove all doubt.

## Exercises

**1.** A video is 7,384 seconds long. Print it as hours, minutes and seconds.

<details>
<summary>Show solution</summary>

```python
total = 7384
hours = total // 3600
minutes = (total % 3600) // 60
seconds = total % 60
print(hours, "h", minutes, "m", seconds, "s")   # 2 h 3 m 4 s
```

</details>

**2.** Write one expression that is `True` when `year` is a leap year: divisible by 4, and either not divisible by 100 or divisible by 400.

<details>
<summary>Show solution</summary>

```python
year = 2024
is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
print(is_leap)   # True
```

</details>

**3.** Predict each result, then check: `10 / 5`, `10 // 3`, `-10 // 3`, `2 ** 3 ** 2`.

<details>
<summary>Show solution</summary>

`2.0`, `3`, `-4` and `512`. The last one surprises people: `**` groups right to left, so it is `2 ** (3 ** 2)`, which is `2 ** 9`.

</details>

```quiz
[
  {
    "q": "What is `17 % 5`?",
    "options": ["`3`", "`2`", "`3.4`", "`12`"],
    "answer": 1,
    "why": "17 divided by 5 is 3 with remainder 2. Modulo returns the remainder."
  },
  {
    "q": "What does `8 / 2` return?",
    "options": ["`4`", "`4.0`", "`\"4\"`", "It depends on the Python version"],
    "answer": 1,
    "why": "True division always returns a float in Python 3. Use `//` for an integer result."
  },
  {
    "q": "In `x != 0 and 10 / x > 2`, what happens when `x` is `0`?",
    "options": [
      "A `ZeroDivisionError` is raised",
      "The expression is `False` and the division never runs",
      "The expression is `True`",
      "Python asks for a new value"
    ],
    "answer": 1,
    "why": "`and` short-circuits. The left side is false, so Python never evaluates the division."
  }
]
```
