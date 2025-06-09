# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import xml.etree.ElementTree as ET
import hashlib


def hash_nonlinearity(xml_str):
    def canon(node):
        # canonical repr = tag + sorted(child_reprs) + sorted attributes
        parts = [canon(c) for c in node]
        parts.sort()
        attr = ""
        if node.attrib:
            attr = (
                "{" + ",".join(f"{k}={v}" for k, v in sorted(node.attrib.items())) + "}"
            )
        return node.tag + "".join(parts) + attr

    root = ET.fromstring(xml_str)
    h = hashlib.md5(canon(root).encode("utf-8")).hexdigest()
    return h
