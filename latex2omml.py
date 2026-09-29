# -*- coding: utf-8 -*-
"""latex2omml -- a small, dependency-free LaTeX -> Word OMML converter.

Scope: exactly the math this paper needs (inline runs, sub/superscripts with
brace *and* bare arguments, \\sum/\\prod n-ary operators with limits, \\frac,
delimiters, Greek letters, relations/arrows, operators, spacing, \\text with
mixed CJK, \\mathrm/\\text/\\bigl-style macros).

Why this exists
---------------
The v3.0 master draft carried its display equations as *raw LaTeX source* in a
monospace run (``build_paper_docx.py::eq``).  The delivered manuscript
(``docs/最后一版V1.0.docx``) instead contains real Word equations (``m:oMath``,
Cambria Math, sz 19 half-points) -- those were converted by hand in Word after
the build, so every rebuild silently downgraded the formulas back to source
text.  The reviewer's "公式 (1)-(8) 缺失正文" note makes real formulas a hard
requirement, so the conversion is moved into the build chain here.

Deliberately not a general LaTeX engine: no matrices, no \\left/\\right sizing
arithmetic, no AMS environments.  Unknown macros raise ``LatexError`` rather
than emitting silent garbage.

Usage
-----
    from latex2omml import append_math, LatexError
    append_math(paragraph, r"\\Delta\\mathrm{Acc} = -f p_s + C_{\\max}", "(2)")
"""
from __future__ import annotations

from docx.oxml.ns import qn, nsdecls
from lxml import etree

M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

# Font/size of the math runs, matched to the hand-made equations in V1.0.
MATH_FONT = "Cambria Math"
EA_FONT = "宋体"
MATH_SZ = "19"          # half-points -> 9.5 pt, as in the delivered docx

# Characters that may appear bare as their own token.
_ASCII_OPS = set("+-=<>|,;:!()[]")
_ASCII_CLOSERS = set(")}]")
_ASCII_OPENERS = set("({[")


class LatexError(ValueError):
    """Raised when the input uses LaTeX this converter does not implement."""


# --------------------------------------------------------------------------
# Symbols
# --------------------------------------------------------------------------
GREEK = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ϵ',
    'varepsilon': 'ε', 'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι',
    'kappa': 'κ', 'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π',
    'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ', 'phi': 'ϕ',
    'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
    'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
    'Pi': 'Π', 'Sigma': 'Σ', 'Upsilon': 'Υ', 'Phi': 'Φ', 'Psi': 'Ψ',
    'Omega': 'Ω',
}

SYMBOLS = {
    'cdot': '⋅', 'times': '×', 'le': '≤', 'leq': '≤', 'ge': '≥',
    'geq': '≥', 'ne': '≠', 'neq': '≠', 'approx': '≈', 'equiv': '≡',
    'cong': '≅', 'sim': '∼', 'propto': '∝', 'in': '∈', 'notin': '∉',
    'subset': '⊂', 'subseteq': '⊆', 'cup': '∪', 'cap': '∩', 'emptyset': '∅',
    'forall': '∀', 'exists': '∃', 'neg': '¬', 'land': '∧', 'lor': '∨',
    'pm': '±', 'mp': '∓', 'infty': '∞', 'partial': '∂', 'nabla': '∇',
    'to': '→', 'rightarrow': '→', 'leftarrow': '←', 'mapsto': '↦',
    'Rightarrow': '⇒', 'Longrightarrow': '⟹', 'Leftrightarrow': '⇔',
    'ldots': '…', 'dots': '…', 'cdots': '⋯', 'ast': '∗', 'star': '⋆',
    'circ': '∘', 'prime': '′', 'angle': '∠', 'dagger': '†',
}

# Commands that print as upright multi-letter operators.
UPRIGHT = {
    'log': 'log', 'ln': 'ln', 'exp': 'exp', 'sin': 'sin', 'cos': 'cos',
    'tan': 'tan', 'min': 'min', 'max': 'max', 'arg': 'arg', 'deg': 'deg',
    'dim': 'dim', 'det': 'det', 'inf': 'inf', 'sup': 'sup', 'lim': 'lim',
}

# \Pr renders upright; Word has no dedicated operator, so use the letters.
LITERAL_WORDS = {
    'Pr': 'Pr',
}

BIG_OPS = {'sum': '∑', 'prod': '∏', 'int': '∫', 'bigcup': '⋃', 'bigcap': '⋂'}

# Macros that produce an opening + closing delimiter pair.
DELIM_PAIRS = {
    'bigl': ('(', ')'), 'bigr': (None, ')'), 'Bigl': ('(', ')'),
    'Bigr': (None, ')'), 'biggl': ('(', ')'), 'biggr': (None, ')'),
    'left': ('(', ')'), 'right': (None, ')'),
    'lVert': ('‖', '‖'), 'rVert': ('‖', '‖'), 'Vert': ('‖', '‖'),
    'lvert': ('|', '|'), 'rvert': ('|', '|'), 'lvert*': ('|', '|'),
}

# Absolute-value / norm helpers: \bigl| ... \bigr|
DELIM_BARS = {'bigl|': '|', 'bigr|': '|', 'Bigl|': '|', 'Bigr|': '|',
              'lvert': '|', 'rvert': '|', 'left|': '|', 'right|': '|',
              'lVert': '‖', 'rVert': '‖', 'Vert': '‖'}

# Delimiter-sizing macros take the delimiter as their argument, e.g.
#   \bigl( ... \bigr)     \bigl[ ... \bigr]     \bigl| ... \bigr|
#   \left( ... \right)    \left. (invisible)
# The tokenizer glues the delimiter onto the macro so the parser sees a
# single token such as 'bigl('.
DELIM_SIZERS = {'bigl', 'bigr', 'Bigl', 'Bigr', 'biggl', 'biggr',
                'Biggl', 'Biggr', 'left', 'right'}

# Character a glued delimiter-sizing token maps to ('.' = invisible).
DELIM_CHARS = {'(': '(', ')': ')', '[': '[', ']': ']', '|': '|'}

# Spacing commands -> em widths.
SPACES = {
    ',': 0.167, ';': 0.278, ':': 0.222, '!': -0.167, ' ': 0.25,
    'thinspace': 0.167, 'medspace': 0.222, 'thickspace': 0.278,
    'enspace': 0.5, 'quad': 1.0, 'qquad': 2.0,
}

# Accents / decorations.
ACCENTS = {
    'hat': '̂', 'widehat': '̂', 'bar': '̄', 'overline': '̄',
    'tilde': '̃', 'widetilde': '̃', 'dot': '̇', 'ddot': '̈',
    'vec': '⃗',
}

FUNCTIONS = {'frac': 2, 'dfrac': 2, 'tfrac': 2, 'sqrt': 1,
             'text': 1, 'mathrm': 1, 'operatorname': 1, 'mbox': 1,
             'mathsf': 1, 'mathtt': 1, 'mathbf': 1, 'mathit': 1,
             'overline': 1, 'underline': 1, 'hat': 1, 'widehat': 1,
             'bar': 1, 'tilde': 1, 'widetilde': 1, 'dot': 1, 'ddot': 1,
             'vec': 1, 'substack': 1}

# Commands that are pure no-ops for our purposes.
IGNORED = {'nolimits', 'limits', 'displaystyle', 'textstyle',
           'scriptstyle', 'allowbreak', 'nonumber', 'notag', 'big'}

SUBSCRIPT_WORDS = {'max': 'max', 'min': 'min', 'out': 'out', 'c': 'c',
                   's': 's', 't': 't', 'P': 'P', 'Q': 'Q', 'r': 'r',
                   'g': 'g', 'e': 'e', 'i': 'i', 'j': 'j', 'k': 'k',
                   '0': '0', '1': '1', 'th': 'th'}


def _strip_math_space_escapes(s: str) -> str:
    """Remove ``\\ `` (escaped space) which appears in some r-strings."""
    return s.replace('\\ ', '\u00a0')


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
def tokenize(src: str) -> list[tuple[str, str]]:
    """Return ``[(kind, text), ...]`` with kind in {ctrl, char, group, space}."""
    src = _strip_math_space_escapes(src)
    toks: list[tuple[str, str]] = []
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch == '\\':
            m = i + 1
            while m < n and src[m].isalpha():
                m += 1
            if m > i + 1:
                name = src[i + 1:m]
                # A control word ends at the first non-alpha character.
                # An earlier version also glued a following char from
                # '|{}[],;:! ' into the name, which broke two common cases:
                #   \Delta Acc   -> control word 'Delta ' (unknown command)
                #   \mathrm{Acc} -> control word 'mathrm{' (stray '}')
                # Non-alpha control symbols such as \, \; \| \{ are already
                # handled by the else-branch below.  The one glueing rule we
                # do need is delimiter sizing: \bigl( \bigr) \left[ \right.
                if name in DELIM_SIZERS:
                    k = m
                    while k < n and src[k] == ' ':
                        k += 1
                    if k < n and src[k] in DELIM_CHARS or \
                            (k < n and src[k] == '.'):
                        name += src[k]
                        m = k + 1
                toks.append(('ctrl', name))
                i = m
            else:
                nxt = src[i + 1] if i + 1 < n else ''
                toks.append(('ctrl', nxt))
                i += 2
            continue
        if ch == '{':
            depth, j = 1, i + 1
            while j < n and depth:
                if src[j] == '\\':
                    j += 2
                    continue
                if src[j] == '{':
                    depth += 1
                elif src[j] == '}':
                    depth -= 1
                j += 1
            if depth:
                raise LatexError('unbalanced braces in %r' % src)
            toks.append(('group', src[i + 1:j - 1]))
            i = j
            continue
        if ch == '}':
            raise LatexError('unexpected } in %r' % src)
        if ch.isspace():
            toks.append(('space', ' '))
            i += 1
            continue
        if ch == '_':
            toks.append(('char', '_'))
            i += 1
            continue
        if ch == '^':
            toks.append(('char', '^'))
            i += 1
            continue
        if ch == '&':
            toks.append(('char', '&'))
            i += 1
            continue
        # XML-escaped relations may appear literally from the source strings.
        if src.startswith('&gt;', i):
            toks.append(('char', '>'))
            i += 4
            continue
        if src.startswith('&lt;', i):
            toks.append(('char', '<'))
            i += 4
            continue
        if src.startswith('&amp;', i):
            toks.append(('char', '&'))
            i += 5
            continue
        toks.append(('char', ch))
        i += 1
    return toks


# --------------------------------------------------------------------------
# Node model
# --------------------------------------------------------------------------
# node := ('r', text)                       ordinary math run
#      |  ('s', base, sub, sup)             subscript / superscript
#      |  ('n', chr, sub, sup)              n-ary operator
#      |  ('f', num_nodes, den_nodes)       fraction
#      |  ('b', base_nodes)                 (reserved: sqrt/overline)
#      |  ('a', accent_char, base)          accent
#      |  ('sp', em)                        spacing


def parse(toks: list[tuple[str, str]]) -> list:
    nodes, pos = _parse_seq(toks, 0)
    if pos != len(toks):
        raise LatexError('trailing tokens at %d: %r' % (pos, toks[pos:pos + 3]))
    return nodes


def _parse_seq(toks, pos, stop_at: set[str] | None = None) -> tuple[list, int]:
    # Recursive call sites pass the *raw body* of a group (a plain string),
    # e.g. the argument of \mathrm{...} or a script argument.  Accept that
    # here and tokenize, rather than making every caller remember to.  Before
    # this guard, _parse_seq('\\max', 0) iterated over individual characters
    # and died with "not enough values to unpack (expected 2, got 1)".
    if isinstance(toks, str):
        toks = tokenize(toks)
        pos = 0
    nodes: list = []
    while pos < len(toks):
        kind, text = toks[pos]
        if kind == 'space':
            pos += 1
            continue
        if kind == 'char' and text in _ASCII_CLOSERS:
            if stop_at and text in stop_at:
                return nodes, pos
            # Bare ) ] } outside a paired delimiter are ordinary characters,
            # e.g. the superscript in R_s^{(0)}.  Emit literally instead of
            # rejecting (an earlier version raised here).
            nodes.append(('r', text))
            pos += 1
            continue
        if kind == 'char' and text in _ASCII_OPENERS:
            # Bare ( [ { are ordinary characters in LaTeX math (they are only
            # delimiters when paired).  The previous version rejected them
            # outright, which made standard notation such as the superscript
            # in R_s^{(0)} impossible to express.  Emit them literally; use
            # \bigl( ... \bigr) or \left( ... \right) when real paired
            # delimiters are wanted.
            nodes.append(('r', text))
            pos += 1
            continue
        if kind == 'ctrl' and len(text) > 1 and text[:-1] in DELIM_SIZERS:
            # Glued delimiter-sizing token, e.g. 'bigl(' / 'bigr)' / 'left['.
            sym = DELIM_CHARS.get(text[-1])
            if sym:
                nodes.append(('r', sym))
            pos += 1
            continue
        if kind == 'char' and text == '&':
            nodes.append(('sp', 0.278))
            pos += 1
            continue
        if kind == 'ctrl' and text.endswith('|'):
            nodes.append(('r', '|'))
            pos += 1
            continue
        if kind == 'ctrl' and text in DELIM_PAIRS and text not in ('left',
                                                                   'right'):
            nodes.append(('r', DELIM_PAIRS[text][0]))
            pos += 1
            continue
        if kind == 'ctrl' and text in ('left', 'right', 'bigl', 'bigr',
                                       'Bigl', 'Bigr', 'biggl', 'biggr'):
            pos += 1
            if pos >= len(toks):
                raise LatexError('%s without a delimiter' % text)
            k2, t2 = toks[pos]
            if t2 == '(':
                nodes.append(('r', '('))
            elif t2 == ')':
                nodes.append(('r', ')'))
            elif t2 in ('|', 'vert', 'rvert', 'lvert', 'Vert', 'rVert',
                        'lVert'):
                nodes.append(('r', '|' if 'l' in t2 or 'r' in t2 else '|'))
            elif t2 == '.':
                pass
            else:
                raise LatexError('unsupported delimiter %r after %s'
                                 % (t2, text))
            pos += 1
            continue
        if kind == 'ctrl' and text in IGNORED:
            pos += 1
            continue
        if kind == 'ctrl' and text in SPACES:
            nodes.append(('sp', SPACES[text]))
            pos += 1
            continue
        if kind == 'ctrl' and text in BIG_OPS:
            pos += 1
            sub_s, pos = _maybe_arg(toks, pos)
            sup_s, pos = _maybe_arg(toks, pos)
            # _maybe_arg returns the RAW argument text; parse it into nodes
            # here.  Storing the raw string made _write_script iterate over
            # its characters and die with "unknown node kind 'c'".
            sub = _parse_seq(sub_s, 0)[0] if sub_s else None
            sup = _parse_seq(sup_s, 0)[0] if sup_s else None
            nodes.append(('n', BIG_OPS[text], sub, sup))
            continue
        if kind == 'ctrl' and text in ('frac', 'dfrac', 'tfrac'):
            pos += 1
            num, pos = _require_group(toks, pos, text)
            den, pos = _require_group(toks, pos, text)
            nodes.append(('f', _parse_seq(num, 0)[0], _parse_seq(den, 0)[0]))
            continue
        if kind == 'ctrl' and text == 'sqrt':
            raise LatexError('\\sqrt not needed by this paper')
        if kind == 'ctrl' and text in ('text', 'mathrm', 'operatorname',
                                       'mbox', 'mathsf', 'mathtt', 'mathbf',
                                       'mathit'):
            pos += 1
            body, pos = _require_group(toks, pos, text)
            letters = body
            if text in ('operatorname', 'mathrm', 'mathbf'):
                letters = _split_upright(body)
            else:
                letters = body.replace('\\ ', '\u00a0')
            base = [('r', ' ' if ch == ' ' else ch) for ch in letters]
            attached, pos = _attach_scripts(toks, pos, base)
            nodes.extend(attached)
            continue
        if kind == 'ctrl' and text in ACCENTS:
            pos += 1
            body, pos = _require_group(toks, pos, text)
            nodes.append(('a', ACCENTS[text], _parse_seq(body, 0)[0]))
            continue
        if kind == 'ctrl' and text == 'overline':
            pos += 1
            body, pos = _require_group(toks, pos, text)
            nodes.append(('a', '̄', _parse_seq(body, 0)[0]))
            continue
        if kind == 'ctrl' and text in GREEK:
            nodes.append(('r', GREEK[text]))
            pos += 1
            continue
        if kind == 'ctrl' and text in SYMBOLS:
            nodes.append(('r', SYMBOLS[text]))
            pos += 1
            continue
        if kind == 'ctrl' and text in UPRIGHT:
            nodes.append(('r', UPRIGHT[text]))
            pos += 1
            continue
        if kind == 'ctrl' and text in LITERAL_WORDS:
            attached, pos = _attach_scripts(
                toks, pos + 1, [('r', LITERAL_WORDS[text])])
            nodes.extend(attached)
            continue
        if kind == 'ctrl':
            raise LatexError('unsupported command \\%s' % text)
        # Ord atom: char or group, then optional scripts.
        base, pos = _parse_atom(toks, pos)
        sub = sup = None
        while True:
            pos = _skip_ignored(toks, pos)
            if not (pos < len(toks) and toks[pos][0] == 'char'
                    and toks[pos][1] in ('_', '^')):
                break
            which = toks[pos][1]
            pos += 1
            arg, pos = _require_arg(toks, pos, 'script')
            arg_nodes = _parse_seq(arg, 0)[0]
            if which == '_':
                sub = (sub or []) + arg_nodes
            else:
                sup = (sup or []) + arg_nodes
        if sub is not None or sup is not None:
            nodes.append(('s', base, sub, sup))
        else:
            nodes.extend(base)
    return nodes, pos


def _parse_atom(toks, pos) -> tuple[list, int]:
    kind, text = toks[pos]
    if kind == 'group':
        return _parse_seq(text, 0)[0], pos + 1
    if kind == 'char' and text in _ASCII_OPS:
        return [('r', text)], pos + 1
    if kind == 'char' and text == "_":
        raise LatexError("subscript without a base")
    if kind == 'char' and text == "^":
        raise LatexError("superscript without a base")
    if kind == 'ctrl':
        raise LatexError('unexpected command at atom position: \\%s' % text)
    return [('r', text)], pos + 1


def _maybe_arg(toks, pos):
    if pos < len(toks) and toks[pos][0] == 'char' and toks[pos][1] in ('_', '^'):
        pos += 1
        return _require_arg(toks, pos, 'script')
    return None, pos


def _skip_ignored(toks, pos):
    """Advance past no-op commands (\\nolimits, \\displaystyle, ...).

    They must not break script attachment: ``\\Pr\\nolimits_{P}`` otherwise
    leaves the '_' orphaned and raises "subscript without a base".
    """
    while (pos < len(toks) and toks[pos][0] == 'ctrl'
           and toks[pos][1] in IGNORED):
        pos += 1
    return pos


def _attach_scripts(toks, pos, base_nodes):
    """Attach any following _/^ scripts to an already-parsed base.

    Needed for bases that are emitted by their own branch (multi-letter
    upright groups such as \\mathrm{Pr}, and LITERAL_WORDS): those branches
    ``continue`` past the generic Ord-atom path that normally consumes
    scripts, so without this the following '_' hit _parse_atom bare and
    raised "subscript without a base".
    """
    sub = sup = None
    while True:
        pos = _skip_ignored(toks, pos)
        if not (pos < len(toks) and toks[pos][0] == 'char'
                and toks[pos][1] in ('_', '^')):
            break
        which = toks[pos][1]
        pos += 1
        arg, pos = _require_arg(toks, pos, 'script')
        arg_nodes = _parse_seq(arg, 0)[0]
        if which == '_':
            sub = (sub or []) + arg_nodes
        else:
            sup = (sup or []) + arg_nodes
    if sub is not None or sup is not None:
        return [('s', base_nodes, sub, sup)], pos
    return base_nodes, pos


def _require_arg(toks, pos, what) -> tuple[str, int]:
    if pos >= len(toks):
        raise LatexError('%s argument missing' % what)
    kind, text = toks[pos]
    if kind == 'space':
        pos += 1
        if pos >= len(toks):
            raise LatexError('%s argument missing' % what)
        kind, text = toks[pos]
    if kind == 'group':
        return text, pos + 1
    if kind == 'char':
        return text, pos + 1
    if kind == 'ctrl':
        return '\\' + text, pos + 1
    raise LatexError('%s argument malformed' % what)


def _require_group(toks, pos, cmd) -> tuple[str, int]:
    if pos >= len(toks):
        raise LatexError('\\%s needs a braced argument' % cmd)
    kind, text = toks[pos]
    if kind == 'group':
        return text, pos + 1
    raise LatexError('\\%s needs a braced argument, got %r' % (cmd, text))


def _split_upright(body: str) -> str:
    """Handle nested simple macros inside \\mathrm{...} bodies."""
    out = []
    i = 0
    while i < len(body):
        if body[i] == '\\':
            j = i + 1
            while j < len(body) and body[j].isalpha():
                j += 1
            name = body[i + 1:j]
            if name in GREEK:
                out.append(GREEK[name])
            elif name in SYMBOLS:
                out.append(SYMBOLS[name])
            elif name in UPRIGHT:
                out.append(UPRIGHT[name])
            else:
                out.append(name)
            i = j
            continue
        out.append(body[i])
        i += 1
    return ''.join(out)


# --------------------------------------------------------------------------
# OMML writer
# --------------------------------------------------------------------------
def _sub(tag: str):
    return etree.SubElement


def _el(parent, tag: str):
    return etree.SubElement(parent, '{%s}%s' % (M_NS, tag))


def _run_props(parent):
    rPr = etree.SubElement(parent, '{%s}rPr' % W_NS)
    fonts = etree.SubElement(rPr, '{%s}rFonts' % W_NS)
    fonts.set('{%s}hint' % W_NS, 'default')
    fonts.set('{%s}ascii' % W_NS, MATH_FONT)
    fonts.set('{%s}hAnsi' % W_NS, MATH_FONT)
    fonts.set('{%s}eastAsia' % W_NS, EA_FONT)
    sz = etree.SubElement(rPr, '{%s}sz' % W_NS)
    sz.set('{%s}val' % W_NS, MATH_SZ)
    return rPr


def _mrun(parent, text: str):
    r = _el(parent, 'r')
    _el(r, 'rPr')
    _run_props(r)
    t = _el(r, 't')
    t.text = text
    return r


def _write_script(parent, tag: str, nodes):
    """Write m:sub / m:sup. ``nodes`` None means an empty slot (skipped)."""
    if not nodes:
        return
    box = _el(parent, tag)
    for node in nodes:
        _write_node(box, node)


def _write_node(parent, node):
    kind = node[0]
    if kind == 'r':
        _mrun(parent, node[1])
    elif kind == 'sp':
        _mrun(parent, '\u2009' * max(1, int(round(node[1] * 4))))
    elif kind == 's':
        _, base, sub, sup = node
        ssub = _el(parent, 'sSub')
        _el(ssub, 'sSubPr')
        e = _el(ssub, 'e')
        for n in base:
            _write_node(e, n)
        _write_script(ssub, 'sub', sub)
        _write_script(ssub, 'sup', sup)
    elif kind == 'n':
        _, char, sub, sup = node
        nary = _el(parent, 'nary')
        pr = _el(nary, 'naryPr')
        chr_el = _el(pr, 'chr')
        chr_el.set('{%s}val' % M_NS, char)
        lim = _el(pr, 'limLoc')
        lim.set('{%s}val' % M_NS, 'subSup')
        if not sup:
            hide = _el(pr, 'supHide')
            hide.set('{%s}val' % M_NS, '1')
        _write_script(nary, 'sub', sub)
        _write_script(nary, 'sup', sup)
        _el(nary, 'e')
    elif kind == 'f':
        _, num, den = node
        frac = _el(parent, 'f')
        _el(frac, 'fPr')
        nbox = _el(frac, 'num')
        for n in num:
            _write_node(nbox, n)
        dbox = _el(frac, 'den')
        for n in den:
            _write_node(dbox, n)
    elif kind == 'a':
        _, accent, base = node
        acc = _el(parent, 'acc')
        pr = _el(acc, 'accPr')
        chr_el = _el(pr, 'chr')
        chr_el.set('{%s}val' % M_NS, accent)
        e = _el(acc, 'e')
        for n in base:
            _write_node(e, n)
    else:  # pragma: no cover
        raise LatexError('unknown node kind %r' % (kind,))


def latex_to_omml(latex: str):
    """Return an ``m:oMath`` lxml element for ``latex``."""
    math = etree.Element('{%s}oMath' % M_NS, nsmap={'m': M_NS})
    for node in parse(tokenize(latex)):
        _write_node(math, node)
    return math


def append_math(paragraph, latex: str, label: str | None = None):
    """Append ``latex`` (as a real Word equation) and an optional label run.

    Mirrors the structure of the equations that were converted by hand in the
    delivered v3.0/V1.0 docx: ``m:oMath`` followed by a Cambria Math run
    carrying the equation number, both inside the centred paragraph.
    """
    math = latex_to_omml(latex)
    paragraph._p.append(math)
    if label:
        run = paragraph.add_run('\u2009' * 4 + label)
        rPr = run._r.get_or_add_rPr()
        fonts = rPr.find(qn('w:rFonts'))
        if fonts is None:
            fonts = rPr.makeelement(qn('w:rFonts'), {})
            rPr.insert(0, fonts)
        fonts.set(qn('w:ascii'), MATH_FONT)
        fonts.set(qn('w:hAnsi'), MATH_FONT)
        fonts.set(qn('w:eastAsia'), EA_FONT)
        sz = rPr.makeelement(qn('w:sz'), {})
        sz.set(qn('w:val'), MATH_SZ)
        rPr.append(sz)
    return paragraph


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
_EQUATIONS = [
    (r"\mathrm{Acc} \;=\; \sum_{c} p_c\, R_c", "(1)"),
    (r"\Delta\mathrm{Acc} \;=\; -\, f\, p_s \;+\; \mathrm{collateral},"
     r" \qquad \mathrm{collateral} := \sum_{c\neq s} p_c\, \Delta R_c", "(2)"),
    (r"\bigl|\mathrm{collateral}\bigr| \;\le\; \sum_{c\neq s} p_c\,"
     r" \bigl|\Delta R_c\bigr|", "(3)"),
    (r"C_{\max} \;:=\; \sum_{c\neq s} p_c\, \bigl(1 - R_c^{(0)}\bigr)", "(4)"),
    (r"R_s' \le \delta \;\Longrightarrow\; \Delta\mathrm{Acc} \;\le\;"
     r" -\,\bigl(R_s^{(0)} - \delta\bigr)\, p_s \;+\; C_{\max}", "(5)"),
    (r"\bigl(R_s^{(0)} - \delta\bigr)\, p_s \;>\; C_{\max} + \varepsilon"
     r" \;\Longrightarrow\; \bigl|\Delta\mathrm{Acc}\bigr| > \varepsilon", "(6)"),
    (r"\bigl|\Delta\mathrm{Acc}\bigr| \;\ge\; \bigl(R_s^{(0)} - "
     r"\delta\bigr)\, p_s \;-\; C_{\max}", "(7)"),
    (r"\bigl|\Pr\nolimits_{P}[A] - \Pr\nolimits_{Q}[A]\bigr|"
     r" \;\le\; \mathrm{TV}(P, Q)", "(8)"),
    (r"\mathrm{Acc} = \sum_{c} \frac{\mathrm{TP}_c}{|T|}"
     r" = \sum_{c} \frac{|T_c|}{|T|}\,\frac{\mathrm{TP}_c}{|T_c|}"
     r" = \sum_{c} p_c\, R_c", None),
    (r"D:\ (P',\, G_t)\ \mapsto\ P_{\mathrm{out}}"
     r" \qquad \text{（确定性）}", None),
]


def main() -> None:
    ok = 0
    for latex, label in _EQUATIONS:
        try:
            el = latex_to_omml(latex)
            xml = etree.tostring(el, pretty_print=False, encoding='unicode')
            import re
            text = ''.join(re.findall(r'<m:t[^>]*>([^<]*)</m:t>', xml))
            print('[OK ] %-12s %s' % (label or '-', text[:90]))
            ok += 1
        except LatexError as exc:
            print('[ERR] %-12s %s' % (label or '-', exc))
    print('\n%d/%d equations converted' % (ok, len(_EQUATIONS)))


if __name__ == '__main__':
    main()
