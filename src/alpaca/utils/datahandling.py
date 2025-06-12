# -*- coding: utf-8 -*-
"""
@authors: kuen,
"""
import xml.etree.ElementTree as ET
import zlib


def hash_nonlinearity(xml_str):
    """Convert XML string to a short 8-character hash."""

    def canon(node):
        """Generate canonical representation: tag + sorted children + sorted attributes"""
        parts = [canon(c) for c in node]
        parts.sort()
        attrs = "".join(f"{k}={v}" for k, v in sorted(node.attrib.items()))
        return node.tag + "".join(parts) + attrs

    root = ET.fromstring(xml_str)
    canonical = canon(root).encode("utf-8")
    crc = zlib.crc32(canonical) & 0xFFFFFFFF  # Ensure unsigned 32-bit
    return f"{crc:08x}"  # Format as 8-digit hex
