"""Schema to automaton, and the boundary where a DFA is no longer enough.

The corpus's distinction is the load-bearing one. A finite automaton "doesn't
have any way of keeping track of how many times it's been in a state before",
so arbitrary nesting -- "match numbers of a's and b's or match numbers of
parentheses… or json max numbers of curly braces" -- needs a pushdown
automaton, "a stack of prior values". Fixed nesting can be flattened into a
DFA by giving each depth its own state ("this is the state for brackets nested
five times versus six versus seven"), and that is exactly why it explodes.

And the claim the design turns on: "in general, anything that supports uh JSON
schemas is actually writing push down automa to enforce its constraints, not
FSAs".

Everything here is deliberately tiny so the state counts are inspectable by
hand. The counts are this model's; no real schema is compiled.
"""

# --------------------------------------------------------------- DFA core


class DFA:
    """States are ints; `transitions[state][symbol] = next_state`.

    `accept` is a set of states. A missing symbol means the transition is
    illegal, which is precisely the information the mask is built from.
    """

    def __init__(self, transitions, accept, start=0):
        self.transitions = transitions
        self.accept = set(accept)
        self.start = start

    @property
    def n_states(self):
        return len(self.transitions)

    def legal_symbols(self, state, alphabet):
        row = self.transitions.get(state, {})
        return [s for s in alphabet if s in row]

    def step(self, state, symbol):
        return self.transitions.get(state, {}).get(symbol)

    def accepts(self, tokens):
        state = self.start
        for t in tokens:
            state = self.step(state, t)
            if state is None:
                return False
        return state in self.accept

    def live_states(self):
        """States reachable from start. Dead states are grammar bugs."""
        seen, stack = {self.start}, [self.start]
        while stack:
            s = stack.pop()
            for nxt in self.transitions.get(s, {}).values():
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen


# ------------------------------------------------- bracket language: {}^n

def bracket_fsa(max_depth):
    """Accepts `{`^n `}`^n for n <= max_depth. State count grows with depth.

    This is the flattening the corpus describes: the automaton has to *be* at
    a depth rather than remember one, so depth k needs its own state. State k
    means "k unmatched braces"; state 0 is both the start and the accept
    state. max_depth + 1 states.
    """
    transitions = {}
    for k in range(max_depth + 1):
        row = {}
        if k < max_depth:
            row["{"] = k + 1
        if k >= 1:
            row["}"] = k - 1
        transitions[k] = row
    return DFA(transitions, accept={0})


class CounterPDA:
    """Accepts `{`^n `}`^n for ANY n, with one state and one stack.

    This is the thing the DFA cannot be: the memory is unbounded, so no finite
    state set can stand in for it.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    @property
    def n_states(self):
        return 1

    def accepts(self, tokens):
        depth = 0
        for t in tokens:
            if t == "{":
                depth += 1
                if self.max_depth is not None and depth > self.max_depth:
                    return False
            elif t == "}":
                depth -= 1
                if depth < 0:
                    return False
            else:
                return False
        return depth == 0


# ------------------------------------------------- JSON-ish schema -> DFA

# The toy vocabulary. Keys and values are single tokens, which is a
# simplification stated in the prose: real tokenizers split `"name"` across
# several tokens, and that is the multi-token-code hazard the LLD tracks.
STRUCTURAL = [
    "{", "}", "\"name\"", "\"birth_year\"", ":", ",", "Taylor", "Swift",
    "1989", "2001",
]
N_FILLER = 2048
VOCAB = STRUCTURAL + ["f%04d" % i for i in range(N_FILLER)]
VID = {tok: i for i, tok in enumerate(VOCAB)}


def json_schema_fsa(fields):
    """Compile a flat schema into a DFA over VOCAB, all fields required.

    `fields` is a list of (key_token, [value_tokens]). Layout, four states per
    field plus a start and an accept:

        0        expect '{'
        K_i      expect key i          K_i = 1 + 4i
        A_i      expect ':'            A_i = 2 + 4i
        B_i      expect value for i    B_i = 3 + 4i
        C_i      expect ',' or '}'     C_i = 4 + 4i
        ACCEPT   after '}' with none left
    """
    n = len(fields)
    ACCEPT = 1 + 4 * n

    transitions = {0: {"{": 1}}
    for i, (key, values) in enumerate(fields):
        k_i, a_i, b_i, c_i = 1 + 4 * i, 2 + 4 * i, 3 + 4 * i, 4 + 4 * i
        transitions[k_i] = {key: a_i}
        transitions[a_i] = {":": b_i}
        transitions[b_i] = {v: c_i for v in values}
        transitions[c_i] = (
            {",": 1 + 4 * (i + 1)} if i + 1 < n else {"}": ACCEPT}
        )
    transitions[ACCEPT] = {}
    return DFA(transitions, accept={ACCEPT}), ACCEPT


def json_schema_fsa_optional(fields):
    """Same schema, but every field may be omitted.

    The automaton must now distinguish "name seen, birth_year not" from the
    reverse, so the key state is keyed by the set of fields still remaining.
    That is the combinatorial cost of flattening optionality into states: the
    key states alone number 2^n.

    States are built as tuples and renumbered at the end, so `n_states` is a
    real count rather than an encoding artefact.
    """
    n = len(fields)
    ACCEPT = ("accept",)
    start = ("key", frozenset(range(n)))

    transitions = {("start",): {"{": start}}
    stack, seen = [start], {start}
    while stack:
        s = stack.pop()
        kind = s[0]
        if kind == "key":
            rem = s[1]
            row = {}
            for i in sorted(rem):
                row[fields[i][0]] = ("colon", i, rem)
            if not rem:
                row["}"] = ACCEPT
            transitions[s] = row
        elif kind == "colon":
            _, i, rem = s
            transitions[s] = {":": ("value", i, rem)}
        elif kind == "value":
            _, i, rem = s
            after = rem - {i}
            transitions[s] = {
                v: (("key", after) if after else ACCEPT)
                for v in fields[i][1]
            }
        else:
            raise AssertionError("unhandled state kind %r" % (kind,))
        for nxt in transitions[s].values():
            if nxt != ACCEPT and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    transitions[ACCEPT] = {}

    used = sorted(k for k in transitions if k != ACCEPT)
    remap = {old: new for new, old in enumerate(used)}
    accept_id = len(used)
    fixed = {}
    for old, row in transitions.items():
        if old == ACCEPT:
            continue
        fixed[remap[old]] = {
            tok: (accept_id if nxt == ACCEPT else remap[nxt])
            for tok, nxt in row.items()
        }
    fixed[accept_id] = {}
    return DFA(fixed, accept={accept_id}), accept_id


# ------------------------------------------------------------- validator

def validate_schema(tokens, fields):
    """The independent check. Deliberately NOT the same code as the DFA.

    The corpus's architecture claim is that the mask and the validator are not
    redundant: the mask guarantees membership in the language, the validator
    guarantees membership in the schema version currently in force. Here the
    validator is written as a direct parser so a stale-compiled-grammar bug
    cannot hide inside shared code.
    """
    i = 0
    if i >= len(tokens) or tokens[i] != "{":
        return False, "expected '{'"
    i += 1
    seen = []
    while True:
        if i >= len(tokens):
            return False, "truncated"
        tok = tokens[i]
        if tok == "}":
            i += 1
            break
        match = [f for f in fields if f[0] == tok]
        if not match:
            return False, "unknown key %r" % tok
        key, values = match[0]
        seen.append(key)
        i += 1
        if i >= len(tokens) or tokens[i] != ":":
            return False, "expected ':' after %r" % key
        i += 1
        if i >= len(tokens) or tokens[i] not in values:
            return False, "bad value for %r" % key
        i += 1
        if i < len(tokens) and tokens[i] == ",":
            i += 1
            continue
        if i < len(tokens) and tokens[i] == "}":
            i += 1
            break
        return False, "expected ',' or '}'"
    if i != len(tokens):
        return False, "trailing tokens"
    if len(seen) != len(set(seen)):
        return False, "duplicate key"
    return True, "ok"
