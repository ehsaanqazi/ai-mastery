This quiz covers the whole Backend Development module: HTTP and TLS, REST design, JSON, request handling, authentication, authorization, middleware, error handling, documentation, WebSockets and asynchronous APIs. Aim for at least 13 out of 16 before moving on to FastAPI.

```quiz
[
  {
    "q": "Which pair of HTTP methods are both idempotent?",
    "options": ["`POST` and `PUT`", "`PUT` and `DELETE`", "`POST` and `PATCH`", "`PATCH` and `POST`"],
    "answer": 1,
    "why": "Repeating a full `PUT` replacement or a `DELETE` leaves the server in the same state. `POST` is not idempotent and `PATCH` is not guaranteed to be."
  },
  {
    "q": "A response carries `Cache-Control: no-store`. What does that mean?",
    "options": ["Cache it for one hour", "Cache it but revalidate before each use", "No cache, private or shared, may store it", "Only CDNs may store it"],
    "answer": 2,
    "why": "`no-store` forbids storage entirely. `no-cache` is the directive that allows storage with revalidation."
  },
  {
    "q": "What does certificate verification add to a TLS connection?",
    "options": ["Compression", "Assurance that the server is the genuine owner of the requested hostname", "Faster key exchange", "Protection against SQL injection"],
    "answer": 1,
    "why": "Encryption alone could be set up with an impostor. A CA-signed certificate matching the hostname authenticates the server."
  },
  {
    "q": "Why is cursor pagination preferred over offset pagination for large, frequently updated collections?",
    "options": ["Cursors let users jump directly to page 50", "Cursors anchor to the last item seen, so inserts do not cause duplicates or gaps, and indexes find the position directly", "Cursors are encrypted", "Offsets are not allowed in REST"],
    "answer": 1,
    "why": "Offsets count rows that shift on insert and force the database to skip rows. A cursor is a stable position."
  },
  {
    "q": "A client retries `POST /payments` with the same `Idempotency-Key` but a different amount. What should the server do?",
    "options": ["Process the new payment", "Replay the original response", "Reject the request as a misuse of the key", "Delete the original payment"],
    "answer": 2,
    "why": "Reusing a key with a different body is a client bug. Replaying would falsely report the new request as done; rejecting makes the bug visible."
  },
  {
    "q": "What does this print? `print(json.dumps({\"loss\": float(\"nan\")}))`",
    "options": ["`{\"loss\": null}`", "`{\"loss\": NaN}`", "`{\"loss\": \"nan\"}`", "It raises `ValueError`"],
    "answer": 1,
    "why": "Python's default `allow_nan=True` emits the non-standard `NaN` token, which strict parsers reject."
  },
  {
    "q": "A JavaScript client shows ID `9007199254740992` for a record whose ID is `9007199254740993`. What is the fix?",
    "options": ["Use a bigger JSON number", "Send IDs as strings", "Compress the response", "Use XML"],
    "answer": 1,
    "why": "JavaScript numbers are 64-bit floats and cannot represent every integer above 2^53. Strings preserve the exact value."
  },
  {
    "q": "In a server handler, why check `Content-Length` against a limit before reading the body?",
    "options": ["To compute the ETag", "So an oversized body is rejected without loading it into memory", "Because JSON requires it", "To decide the response status code"],
    "answer": 1,
    "why": "Reading first would consume the memory the limit is meant to protect."
  },
  {
    "q": "Which is the correct way to store user passwords?",
    "options": ["`sha256(password)`", "`sha256(salt + password)`", "Argon2id, scrypt or bcrypt with a per-user salt", "Reversible encryption so support can read them"],
    "answer": 2,
    "why": "Password hashes must be salted and deliberately slow. Fast hashes allow billions of guesses per second."
  },
  {
    "q": "What can anyone who obtains a standard signed JWT do without the key?",
    "options": ["Nothing; it is encrypted", "Read all of its claims", "Change its claims and keep the signature valid", "Extend its expiry"],
    "answer": 1,
    "why": "Header and payload are only base64url-encoded. The signature prevents undetected modification, not reading."
  },
  {
    "q": "User A requests `GET /chats/c_77`, which belongs to user B, and receives the chat. What is the vulnerability?",
    "options": ["Cross-site scripting", "Insecure Direct Object Reference (missing object-level authorization)", "Missing CORS headers", "Weak password hashing"],
    "answer": 1,
    "why": "The handler checks that the user is logged in but not that they own the object."
  },
  {
    "q": "A token bucket has capacity 3 and refills 1 token per second. A client sends 5 requests at once from a full bucket. How many succeed?",
    "options": ["1", "3", "5", "0"],
    "answer": 1,
    "why": "The full bucket holds 3 tokens, so 3 requests pass and 2 receive `429` until tokens refill."
  },
  {
    "q": "In RFC 9457 problem details, which member identifies the kind of error for client code to branch on?",
    "options": ["`detail`", "`instance`", "`type`", "`title`"],
    "answer": 2,
    "why": "`type` is a stable URI for the problem kind. `detail` is free text specific to one occurrence."
  },
  {
    "q": "Which change to a public API is breaking?",
    "options": ["Adding an optional request field", "Adding a new endpoint", "Making a previously optional request field required", "Adding a field to a response"],
    "answer": 2,
    "why": "Existing clients that omit the field will start failing validation."
  },
  {
    "q": "A voice assistant streams microphone audio to the server while receiving spoken replies, and the user can interrupt. Which transport fits?",
    "options": ["Polling", "Server-Sent Events", "WebSockets", "Webhooks"],
    "answer": 2,
    "why": "Data flows continuously in both directions at once, which needs a full-duplex connection."
  },
  {
    "q": "A coroutine `f` contains only `time.sleep(1)`. `main` runs `await asyncio.gather(f(), f(), f())`. Roughly how long does `asyncio.run(main())` take?",
    "options": ["About 1 second", "About 3 seconds", "About 0 seconds", "It raises an error"],
    "answer": 1,
    "why": "`time.sleep` blocks the event loop, so the three coroutines run one after another. `await asyncio.sleep(1)` would take about 1 second."
  }
]
```
