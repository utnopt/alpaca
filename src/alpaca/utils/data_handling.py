# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import xml.etree.ElementTree as ET
import zlib
import pyscipopt as scip

from alpaca.utils.localized_string_factory import LocalizedStringFactory as lsf


def hash_nonlinearity(xml_str):
    """Convert XML string to a short 8-character hash."""

    def canon(node):
        """Generate canonical representation: tag + sorted children + sorted attributes"""
        parts = [canon(c) for c in node]
        parts.sort()
        attrs = "".join(f"{k}={v}" for k, v in sorted(node.attrib.items()))
        return node.tag + "".join(parts) + attrs

    root = ET.fromstring(xml_str)
    canonical = canon(root).encode(lsf.file_encoding_utf8())
    crc = zlib.crc32(canonical) & 0xFFFFFFFF  # Ensure unsigned 32-bit
    return f"{crc:08x}"  # Format as 8-digit hex


def pyscipopt_nonlinearity(
    nonlinearity_type: str,
):  # pylint: disable=too-many-return-statements
    """Translate nonlinearity type string to scip function."""
    if nonlinearity_type == lsf.nonlinearity_type_square():
        return lambda x: x**2
    if nonlinearity_type == lsf.nonlinearity_type_exp():
        return scip.exp
    if nonlinearity_type == lsf.nonlinearity_type_ln():
        return scip.log
    if nonlinearity_type == lsf.nonlinearity_type_sqrt():
        return scip.sqrt
    if nonlinearity_type == lsf.nonlinearity_type_sin():
        return scip.sin
    if nonlinearity_type == lsf.nonlinearity_type_cos():
        return scip.cos
    if nonlinearity_type == lsf.nonlinearity_type_log10():
        return lambda x: scip.log(x) / scip.log(10)
    if nonlinearity_type == lsf.nonlinearity_type_tanh():
        return lambda x: (1 - scip.exp(-2 * x)) / (1 + scip.exp(-2 * x))
    if nonlinearity_type == lsf.nonlinearity_type_inverse():
        return lambda x: x**-1
    if nonlinearity_type == lsf.nonlinearity_type_xabsx():
        return abs
    if nonlinearity_type == lsf.nonlinearity_type_negate():
        return lambda x: -x
    return lambda x: x
