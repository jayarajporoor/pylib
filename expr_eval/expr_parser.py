#String expression evaluator
#Copyright Jayaraj Poroor 2020
#Released under MIT License
#
#Expression high-level (abstract) structure:
#
#expr ::= expr '+' expr | expr '-' expr | expr '/' expr | expr '*' expr | identifier | num | '(' expr ')'
#
#Grammar structure:
#
#expr ::= term '+' expr | term '-' expr | term
#term ::= factor '*' expr | factor '/' expr | factor
#factor ::= identifier | num | '(' expr ')'
#

import sys

class ParseError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

def tokenize(input):
    idx = 0
    current_token = ""
    while idx < len(input):
        c = input[idx]
        if c.isspace():
            if current_token != "":
                yield current_token
                current_token = ""
        elif c in ('(', ')', '+', '-', '*', '/'):
            if current_token != "":
                yield current_token
                current_token = ""
            yield c
        else:
            current_token += c
        idx += 1
    if current_token != "":
        yield current_token

class Evaluator:
    def __init__(self, tokenizer=tokenize):
        self.returned_token = None
        self.tokenizer  = tokenizer

    def putback_token(self, token):
        self.returned_token = token

    def next_token(self, tokens, mandatory=False):
        if self.returned_token is not None:
            token = self.returned_token
            self.returned_token = None
            return token
        try:
            token = next(tokens)
            return token
        except StopIteration:
            if mandatory:
                raise ParseError("Unexpected end of input")
            else:
                return None

    def do(self, input):
        tokens = self.tokenizer(input)
        return self.eval_expr(tokens)

    def eval_expr(self, tokens):
        val = self.eval_term(tokens)
        op = self.next_token(tokens)
        if op in ('+', '-'):
                v = self.eval_expr(tokens)
                if op == '+':
                    val = val + v
                else:
                    val = val - v
        else:
            self.putback_token(op)
        return val

    def eval_term(self, tokens):
        val = self.eval_factor(tokens)
        op = self.next_token(tokens)
        if op in ('*', '/'):
            v = self.eval_term(tokens)
            if op == '*':
                val = val * v
            else:
                val = val / v
        elif op is not None:
            self.putback_token(op)
        return val

    def eval_factor(self, tokens):
        token = self.next_token(tokens, mandatory=True)
        if token == '(':
            val = self.eval_expr(tokens)
            token = self.next_token(tokens, mandatory=True)
            if token != ')':
                raise ParseError("Expected ) but found:" + tokens[index])
        else:
            try:
                val = float(token)
            except ValueError:
                raise ParseError("Undefined identifier: " + token)
        return val

if __name__ == "__main__":
    evaluator = Evaluator()
    assert evaluator.do("1+2") == 3
    assert evaluator.do("3*(1+2)") == 9
    print(evaluator.do(sys.argv[1]))
