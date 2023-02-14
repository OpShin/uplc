from rply import LexerGenerator
import re


def strip_comments(s: str):
    ssub = re.sub(r"--.*$", "", s, flags=re.RegexFlag.MULTILINE)
    return ssub


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
    # there may be escaped " inside the string, but not at the end, but there might be an escaped escape at the end
    "TEXT": r'"(([^\n\r"]|\\")*([^\\]|\\\\)|)"',
    "COMMA": r",",
    "DOT": r"\.",
    "NUMBER": r"[-\+]?\d+",
    "NAME": r"[\w_~'][\w\d_~'!#]*",
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


STRING_TOKENS = {
    "BACKSLASH": "\\",
    "CHARESC": r"|".join({"a", "b", "f", "n", "r", "t", "v", '"', "'", "&"}),
    "ASCII": r"|".join(
        {
            "NUL",
            "SOH",
            "STX",
            "ETX",
            "EOT",
            "ENQ",
            "ACK",
            "BEL",
            "BS",
            "HT",
            "LF",
            "VT",
            "FF",
            "CR",
            "SO",
            "SI",
            "DLE",
            "DC1",
            "DC2",
            "DC3",
            "DC4",
            "NAK",
            "SYN",
            "ETB",
            "CAN",
            "EM",
            "SUB",
            "ESC",
            "FS",
            "GS",
            "RS",
            "US",
            "SP",
            "DEL",
        }
    ),
    "CNTRL": r"|".join(set("ABCDEFGHIJKHLMNOPQRSTUVWXYZ") | {"@", "[", "]", "^", "_"}),
    "SPACE": r" ",
    "OTHER_SPACE": r"[\r\n\t \v]",
    "WS": r"\s",
    "GRAPHIC": r"[\d\w]",
}


class StringLexer:
    def __init__(self):
        self.lexer = LexerGenerator()

    def _add_tokens(self):
        for k, v in STRING_TOKENS.items():
            self.lexer.add(k, v)
        self.lexer.ignore("\s+")

    def get_lexer(self):
        self._add_tokens()
        return self.lexer.build()
