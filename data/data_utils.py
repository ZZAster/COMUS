import jieba
import timeout_decorator
from math_tan.math_extractor import MathExtractor


def is_chinese(c):
    return 11904 <= ord(c) <= 42191


def convert_nodes(node):
    # process float number, such as 0.499999999999
    if '.' in node and len(node) >= 10:
        return str(round(float(node), 3))
    # avoid overflow error, -13812983719371929127391283791837918237912837
    if len(node) >= 30:
        return '[UNK]'
    return node


def jieba_retokenize_for_dep(words):
    '''replace formulas with [MATH]'''
    new_words = []
    tag = False
    for word in words:
        if word == '\\fb':
            tag = True
            continue
        elif word == '\\fe':
            tag = False
            new_words.append('[MATH]')
            continue
        if tag or len(word) == 0:
            continue
        if is_chinese(word[0]):
            new_words.extend(list(jieba.cut(word, cut_all=False)))
        else:
            new_words.append(word)

    return new_words


@timeout_decorator.timeout(3)
def formula2opt(f_words):
    '''parsing formulas to operator tree'''
    formula = " ".join(f_words)
    # to address float number, 3 . 4 -> 3.4
    formula = '.'.join(formula.split(' . '))
    root = MathExtractor.parse_from_tex_opt(formula)
    nodes = []
    edges = []

    def add_nodes(curr):
        curr.idx = len(nodes)
        nodes.append(curr.tag.split("!", maxsplit=1)[-1])

        children = curr.children
        if not children or len(children) == 0:
            return
        elif len(children) <= 2:
            for child in children:
                add_nodes(child)
        else:
            raise ValueError("Unexpected children num.")

    def add_edges(curr):
        children = curr.children
        if not children or len(children) == 0:
            return
        for child in children:
            add_edges(child)
            edges.append((curr.idx, child.idx))

    add_nodes(root)
    add_edges(root)

    return nodes, edges


def parse_formula(text):
    new_text = []
    formula_parse = []
    f_words = []
    tag = False
    for word in text:
        if word == "\\fb":
            tag = True
            continue
        elif word == "\\fe":
            tag = False
            try:
                nodes, edges = formula2opt(f_words)
                formula_parse.append((nodes, edges))
                new_text += ['\\fb'] + f_words + ['\\fe']
            except:
                # There may be some formulas that cannot be parsed
                new_text.extend(f_words)
            f_words = []
            continue
        if tag:
            f_words.append(word)
        else:
            new_text.append(word)
    return new_text, formula_parse