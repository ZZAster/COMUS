import re
import io
import xml
from bs4 import BeautifulSoup

from math_tan.semantic_symbol import SemanticSymbol
from sympy.parsing.latex import parse_latex
from sympy.printing.mathml import mathml

__author__ = 'Nidhin, FWTompa, KDavila'


class MathExtractor:
    def __init__(self):
        pass

    namespace = r"(?:[^> :]*:)?"
    attributes = r"(?: [^>]*)?"
    math_expr = "<" + namespace + "math" + attributes + r">.*?</" + namespace + "math>"
    dollars = r"(?<!\\)\$+"
    latex_expr = dollars + ".{1,200}?" + dollars  # converted to math_expr in cleaned text
    # latex could also be surrounded by \(..\) or \[..\], but these are ignored for now (FWT)
    text_token = r"[^<\s]+"

    math_pattern = re.compile(math_expr, re.DOTALL)  # TODO: allow for LaTeX as well
    # split_pattern = re.compile(math_expr+"|"+latex_expr+"|"+text_token, re.DOTALL)

    inner_math = re.compile(".*(<" + math_expr + ")", re.DOTALL)  # rightmost <*:math_tan
    open_tag = re.compile("<(?!/)(?!mws:qvar)" + namespace, re.DOTALL)  # up to and including namespace
    close_tag = re.compile("</(?!mws:qvar)" + namespace, re.DOTALL)  # but keep qvar namespace

    @classmethod
    def isolate_cmml(cls, tree):
        """
        extract the Content MathML from a MathML expr

        param tree: MathML expression
        type  tree: string
        return: Content MathML
        rtype:  string
        """
        parsed_xml = BeautifulSoup(tree, "lxml")

        math_root = parsed_xml.find("math")  # namespaces have been removed (FWT)
        application_tex = math_root.find("annotation", {"encoding": "application/x-tex"})

        if application_tex:
            application_tex.decompose()

        cmml_markup = math_root.find("annotation-xml", {"encoding": "MathML-Content"})
        if cmml_markup:
            cmml_markup.name = "math"
        else:
            cmml_markup = math_root
            pmml_markup = math_root.find("annotation-xml", {"encoding": "MathML-Presentation"})
            if pmml_markup:
                pmml_markup.decompose()  # delete any Content MML

        cmml_markup['xmlns'] = "http://www.w3.org/1998/Math/MathML"  # set the default namespace
        cmml_markup['xmlns:mml'] = "http://www.w3.org/1998/Math/MathML"
        return str(cmml_markup)

    @classmethod
    def convert_to_semanticsymbol(cls, elem):
        """
        Parse expression from Content-MathML

        :param elem: mathml
        :type  elem: string

        :rtype MathSymbol or None
        :return root of symbol tree

        """
        if len(elem) == 0:
            return None

        elem_content = io.StringIO(elem)  # treat the string as if a file
        root = xml.etree.ElementTree.parse(elem_content).getroot()

        return SemanticSymbol.parse_from_mathml(root)

    @classmethod
    def parse_from_tex_opt(cls, tex):
        """
        Parse expression from Tex string using latexmlmath to convert to presentation markup language


        :param tex: tex string
        :type tex string

        :rtype SemanticSymbol
        :return equivalent SemanticSymbol

        """
        cmml = parse_latex(tex)
        cmml = "<math>" + mathml(cmml) + "</math>"
        cmml = cls.isolate_cmml(cmml)
        return cls.convert_to_semanticsymbol(cmml)
