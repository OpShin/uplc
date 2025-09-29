from rply import LexerGenerator
import regex as re

TEXT_REGEX = re.compile(r'"(([^\n\r"]|(?<!\\)(\\\\)*\\")*(?<!\\)(\\\\)*)"')


def strip_comments(s: str):
    # find all occurrences of -- outside of strings and remove everything after it until the end of the line

    changed = True
    while changed:
        all_strings = [x for x in re.finditer(TEXT_REGEX, s)]
        all_comments = reversed(
            list(
                re.finditer(
                    r"--[^\n]*$", s, flags=re.RegexFlag.MULTILINE, overlapped=True
                )
            )
        )
        changed = False
        for comment in all_comments:
            # check if the comment is inside a string
            inside_string = False
            for string in all_strings:
                if string.start() < comment.start() < string.end():
                    inside_string = True
                    break
            if not inside_string:
                # remove the comment
                s = s[: comment.start()] + s[comment.end() :]
                changed = True
                break

    return s


TOKENS = {
    "PROGRAM": r"\bprogram\b",
    "LAMBDA": r"\blam\b",
    "FORCE": r"\bforce\b",
    "DELAY": r"\bdelay\b",
    "BUILTIN": r"\bbuiltin\b",
    "CON": r"\bcon\b",
    "ERROR": r"\berror\b",
    "PAREN_OPEN": r"\(",
    "PAREN_CLOSE": r"\)",
    "BRACK_OPEN": r"\[",
    "BRACK_CLOSE": r"\]",
    "CARET_OPEN": r"\<",
    "CARET_CLOSE": r"\>",
    # there may be an escaped " inside the string (marked by an uneven number of \ before it)
    # the " at the end must be preceded by an even number of \ -- it is not escaped
    "TEXT": TEXT_REGEX.pattern,
    "COMMA": r",",
    "DOT": r"\.",
    "NUMBER": r"[-\+]?\d+",
    "BOOL": r"\b(True|False)\b",
    "I": r"\bI\b",
    "B": r"\bB\b",
    "LIST": r"\bList\b",
    "MAP": r"\bMap\b",
    "CONSTR": r"\bConstr\b",
    "NAME_NON_SPECIAL": r"[\w_~'][\w\d_~'!#]*",
    "HEX": r"#([\dabcdefABCDEF][\dabcdefABCDEF])*",
}


class Lexer:
    def __init__(self):
        self.lexer = LexerGenerator()

    def _add_tokens(self):
        for k, v in TOKENS.items():
            self.lexer.add(k, v)
        self.lexer.ignore("\s+")

    def get_lexer(self):
        self._add_tokens()
        return self.lexer.build()
